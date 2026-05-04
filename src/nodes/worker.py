from langchain_core.messages import HumanMessage, SystemMessage
from src.models.schemas import State, EvidenceItem, Task, Plan
from src.prompts.prompt import WORKER_SYSTEM
from typing import List
from src.services.llm_service import llm
from src.utils.logger import get_logger
import time

# Initialize logger
logger = get_logger("worker")


def worker_node(payload: dict) -> dict:
    start_time = time.time()

    try:
        task = Task(**payload["task"])
        plan = Plan(**payload["plan"])
        evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
        topic = payload["topic"]
        mode = payload.get("mode", "closed_book")

        logger.info(f"Worker started for task ID: {task.id}")
        logger.debug(f"Task title: {task.title}")
        logger.debug(f"Mode: {mode}")
        logger.debug(f"Requires code: {task.requires_code}, Requires citations: {task.requires_citations}")

        bullets_text = "\n- " + "\n- ".join(task.bullets)

        evidence_text = ""
        if evidence:
            logger.debug(f"Evidence items available: {len(evidence)}")

            evidence_text = "\n".join(
                f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
                for e in evidence[:20]
            )

        section_md = llm.invoke(
            [
                SystemMessage(content=WORKER_SYSTEM),
                HumanMessage(
                    content=(
                        f"Blog title: {plan.blog_title}\n"
                        f"Audience: {plan.audience}\n"
                        f"Tone: {plan.tone}\n"
                        f"Blog kind: {plan.blog_kind}\n"
                        f"Constraints: {plan.constraints}\n"
                        f"Topic: {topic}\n"
                        f"Mode: {mode}\n\n"
                        f"Section title: {task.title}\n"
                        f"Goal: {task.goal}\n"
                        f"Target words: {task.target_words}\n"
                        f"Tags: {task.tags}\n"
                        f"requires_research: {task.requires_research}\n"
                        f"requires_citations: {task.requires_citations}\n"
                        f"requires_code: {task.requires_code}\n"
                        f"Bullets:{bullets_text}\n\n"
                        f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                    )
                ),
            ]
        ).content.strip()

        duration = time.time() - start_time

        logger.info(f"Worker completed task ID: {task.id} in {duration:.2f}s")
        logger.debug(f"Generated section length: {len(section_md)} characters")

        return {"sections": [(task.id, section_md)]}

    except Exception as e:
        logger.error(f"Worker failed for task ID {payload.get('task', {}).get('id')}: {str(e)}")
        raise