from __future__ import annotations

import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pandas as pd
import streamlit as st
from PIL import Image

# -----------------------------
# Import your compiled LangGraph app
# -----------------------------
try:
    from src.graphs.main_graph import MainGraph

    app = MainGraph()

except ImportError:
    st.error("Error: Ensure 'src.graphs.main_graph' is accessible.")
    st.stop()

# -----------------------------
# UI Configuration
# -----------------------------
st.set_page_config(
    page_title="BlogeForge AI",
    layout="wide",
    page_icon="✍️"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown(
    """
    <style>

    .main {
        background-color: #f8f9fa;
    }

    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-size: 16px;
    }

    .block-container {
        padding-top: 2rem;
    }

    .workflow-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:

        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists():

            for p in images_dir.rglob("*"):

                if p.is_file():
                    z.write(p, arcname=str(p))

    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:

    current_state = inputs.copy()

    try:

        for step in graph_app.stream(inputs, stream_mode="updates"):

            for node_name, node_output in step.items():

                current_state.update(node_output)

                pretty_name = node_name.replace("_", " ").title()

                if pretty_name not in st.session_state["execution_steps"]:
                    st.session_state["execution_steps"].append(pretty_name)

                yield ("updates", node_name, current_state)

        yield ("final", "complete", current_state)

    except Exception as e:
        st.error(f"Pipeline Error: {e}")


def render_markdown_with_local_images(md: str):

    _MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")

    matches = list(_MD_IMG_RE.finditer(md))

    if not matches:
        st.markdown(md)
        return

    last = 0

    for m in matches:

        st.markdown(md[last:m.start()])

        src = m.group("src").strip().lstrip("./")

        img_path = Path(src)

        if img_path.exists():

            try:
                st.image(
                    Image.open(img_path),
                    use_container_width=True
                )

            except Exception:
                st.caption(f"⚠️ Could not load image: {src}")

        else:
            st.caption(f"🖼️ Missing Image: {src}")

        last = m.end()

    st.markdown(md[last:])


# -----------------------------
# Session State
# -----------------------------
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

if "execution_steps" not in st.session_state:
    st.session_state["execution_steps"] = []

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.title("✍️ BlogeForge")
    st.caption("AI Content Generation Suite")

    st.divider()

    topic = st.text_area(
        "What should the blog be about?",
        placeholder="e.g. The future of AI in 2026...",
        height=150
    )

    with st.expander("⚙️ Advanced Settings"):

        as_of = st.date_input(
            "Research Cutoff Date",
            value=date.today()
        )

        recency = st.slider(
            "Recency (days)",
            1,
            30,
            7
        )

    run_btn = st.button(
        "🚀 Generate Content",
        type="primary"
    )

    st.divider()

    st.subheader("📂 History")

    cwd = Path(".")

    past_files = sorted(
        [p for p in cwd.glob("*.md")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )[:10]

    if past_files:

        selected_file = st.selectbox(
            "Load Recent",
            options=past_files,
            format_func=lambda x: x.name
        )

        if st.button("Load Blog"):

            st.session_state["last_out"] = {
                "final": selected_file.read_text(encoding="utf-8"),
                "plan": {
                    "blog_title": selected_file.stem.replace("_", " ").title(),
                    "tone": "Loaded Draft"
                },
                "evidence": []
            }

# -----------------------------
# Initial Landing Page
# -----------------------------
if not st.session_state["last_out"] and not run_btn:

    st.info(
        """
        ### Getting Started

        1. Enter a topic in the sidebar  
        2. Adjust settings if needed  
        3. Click **Generate Content**
        """
    )

    st.image(
        "https://images.unsplash.com/photo-1455390582262-044cdead277a?auto=format&fit=crop&q=80&w=1200",
        caption="Unleash your creativity."
    )

# -----------------------------
# Generation Logic
# -----------------------------
if run_btn:

    if not topic.strip():

        st.error("Please enter a topic.")

    else:

        st.session_state["execution_steps"] = []

        inputs = {
            "topic": topic.strip(),
            "as_of": as_of.isoformat(),
            "recency_days": recency,
        }

        status_container = st.status(
            "🏗️ Building your blog...",
            expanded=True
        )

        workflow_placeholder = st.empty()

        for kind, node_name, state in try_stream(app, inputs):

            if kind == "updates":

                current_step = node_name.replace("_", " ").title()

                status_container.write(f"✅ {current_step}")

                with workflow_placeholder.container():

                    st.subheader("⚡ Workflow Progress")

                    for i, step in enumerate(
                        st.session_state["execution_steps"],
                        start=1
                    ):
                        st.write(f"{i}. {step}")

            elif kind == "final":

                st.session_state["last_out"] = state

                status_container.update(
                    label="✅ Content Ready!",
                    state="complete",
                    expanded=False
                )

# -----------------------------
# Output Section
# -----------------------------
out = st.session_state.get("last_out")

if out:

    final_md = out.get("final", "")

    plan = out.get("plan", {})

    evidence = out.get("evidence", [])

    title = (
        plan.get("blog_title", "Untitled Blog")
        if isinstance(plan, dict)
        else "Generated Blog"
    )

    slug = safe_slug(title)

    img_dir = Path("images") / slug

    # -----------------------------
    # Header
    # -----------------------------
    st.title(title)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Word Count",
        f"~{len(final_md.split())}"
    )

    col2.metric(
        "Evidence Sources",
        len(evidence)
    )

    col3.metric(
        "Tone",
        str(plan.get("tone", "Professional"))
        if isinstance(plan, dict)
        else "N/A"
    )

    col4.metric(
        "Status",
        "Draft Complete"
    )

    st.divider()

    # -----------------------------
    # Download Buttons
    # -----------------------------
    dl1, dl2, _ = st.columns([1, 1, 2])

    md_filename = f"{slug}.md"

    dl1.download_button(
        label="📄 Download Markdown",
        data=final_md,
        file_name=md_filename,
        mime="text/markdown",
        use_container_width=True,
    )

    bundle = bundle_zip(
        final_md,
        md_filename,
        img_dir
    )

    dl2.download_button(
        label="📦 Download ZIP Bundle",
        data=bundle,
        file_name=f"{slug}_bundle.zip",
        mime="application/zip",
        use_container_width=True,
    )

    # -----------------------------
    # Blog Preview
    # -----------------------------
    st.subheader("📝 Blog Preview")

    with st.container(border=True):

        if final_md:
            render_markdown_with_local_images(final_md)

        else:
            st.warning("No content was generated.")

    st.divider()

    # -----------------------------
    # Evidence Sources
    # -----------------------------
    st.subheader(f"🔎 Research & Evidence ({len(evidence)} sources)")

    if evidence:

        try:

            df_ev = pd.DataFrame(evidence)

            st.dataframe(
                df_ev,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:
            st.error(f"Could not display evidence table: {e}")

    else:
        st.info("No external evidence sources available.")

    st.divider()

    # -----------------------------
    # Generated Images
    # -----------------------------
    st.subheader("🖼️ Generated Images")

    if img_dir.exists():

        imgs = list(img_dir.glob("*"))

        if imgs:

            cols = st.columns(3)

            for i, img_p in enumerate(imgs):

                try:

                    cols[i % 3].image(
                        str(img_p),
                        caption=img_p.name,
                        use_container_width=True
                    )

                except Exception:
                    cols[i % 3].warning(f"Could not load {img_p.name}")

        else:
            st.info("No images found.")

    else:
        st.info("Image generation was not enabled.")