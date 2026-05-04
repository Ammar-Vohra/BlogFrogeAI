from langchain_core.messages import HumanMessage, SystemMessage
from src.models.schemas import State, GlobalImagePlan
from pathlib import Path
from src.prompts.prompt import DECIDE_IMAGES_SYSTEM
from typing import List
from src.services.llm_service import llm
from src.services.openai_service import _openai_generate_image_bytes
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("reducer")


def merge_content(state: State) -> dict:
    logger.info("Merging content started")

    try:
        plan = state["plan"]

        ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
        logger.debug(f"Total sections to merge: {len(ordered_sections)}")

        body = "\n\n".join(ordered_sections).strip()
        merged_md = f"# {plan.blog_title}\n\n{body}"

        logger.info("Content merged successfully")

        return {"merged_md": merged_md}

    except Exception as e:
        logger.error(f"Merge content failed: {str(e)}")
        raise


def decide_images(state: State) -> dict:
    logger.info("Image planning started")

    try:
        planner = llm.with_structured_output(GlobalImagePlan)
        merged_md = state["merged_md"]
        plan = state["plan"]

        assert plan is not None

        logger.debug(f"Blog kind: {plan.blog_kind}")
        logger.debug(f"Content length: {len(merged_md)} characters")

        image_plan = planner.invoke(
            [
                SystemMessage(content=DECIDE_IMAGES_SYSTEM),
                HumanMessage(
                    content=(
                        f"Blog kind: {plan.blog_kind}\n"
                        f"topic: `{state['topic']}`\n"
                        "Insert placeholders + propose image prompts.\n\n"
                        f"{merged_md}"
                    )
                )
            ]
        )

        num_images = len(image_plan.images)
        logger.info(f"Image plan created with {num_images} images")

        return {
            "md_with_placeholders":
                image_plan.md_with_placeholders.strip()
                if image_plan.md_with_placeholders.strip()
                else merged_md,

            "image_specs": [img.model_dump() for img in image_plan.images]
        }

    except Exception as e:
        logger.error(f"Image planning failed: {str(e)}")
        raise


from pathlib import Path

def generate_and_place_images(state: State) -> dict:
    logger.info("Image generation started")

    try:
        plan = state["plan"]
        assert plan is not None

        md_candidate = state.get("md_with_placeholders", "")

        if md_candidate and md_candidate.strip():
            md = md_candidate
        else:
            md = state["merged_md"]

        image_specs = state.get("image_specs", []) or []

        logger.info(f"Total images to process: {len(image_specs)}")

        # -----------------------------
        # Create Output Directories
        # -----------------------------
        base_dir = Path("output")
        images_dir = base_dir / "images"
        blogs_dir = base_dir / "blogs"

        images_dir.mkdir(parents=True, exist_ok=True)
        blogs_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # Safe filename
        # -----------------------------
        safe_name = "".join(
            c for c in plan.blog_title
            if c.isalnum() or c in (" ", "_", "-")
        ).rstrip()

        blog_filename = f"{safe_name}.md"
        blog_path = blogs_dir / blog_filename

        # -----------------------------
        # No images case
        # -----------------------------
        if not image_specs:
            logger.info("No images required, saving markdown directly")
            blog_path.write_text(md, encoding="utf-8")
            logger.info(f"Blog saved at: {blog_path}")
            return {"final": md}

        # -----------------------------
        # Process Images
        # -----------------------------
        for spec in image_specs:
            placeholder = spec["placeholder"]
            img_filename = spec["filename"]
            out_path = images_dir / img_filename

            logger.debug(f"Processing image: {img_filename}")

            if not out_path.exists():
                try:
                    logger.info(f"Generating image: {img_filename}")
                    img_bytes = _openai_generate_image_bytes(spec["prompt"])
                    out_path.write_bytes(img_bytes)

                    logger.info(f"Image saved: {out_path}")

                except Exception as e:
                    logger.error(f"Image generation failed for {img_filename}: {str(e)}")

                    prompt_block = (
                        f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                        f"> **Alt:** {spec.get('alt','')}\n>\n"
                        f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                        f"> **Error:** {e}\n"
                    )

                    md = md.replace(placeholder, prompt_block)
                    continue
            else:
                logger.info(f"Image already exists, skipping: {img_filename}")

            # IMPORTANT: correct relative path from blogs → images
            img_md = f"![{spec['alt']}](../images/{img_filename})\n*{spec['caption']}*"
            md = md.replace(placeholder, img_md)

        # -----------------------------
        # Save final blog
        # -----------------------------
        blog_path.write_text(md, encoding="utf-8")

        logger.info(f"Final blog saved at: {blog_path}")
        logger.info("Image generation completed")

        return {"final": md}

    except Exception as e:
        logger.error(f"Image placement failed: {str(e)}")
        raise