from typing import Optional
from src.graphs.main_graph import MainGraph
from datetime import date, timedelta

app = MainGraph()






def run(topic: str, as_of: Optional[str] = None):
    if as_of is None:
        as_of = date.today().isoformat()

    out = app.invoke(
        {
            "topic": topic,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "as_of": as_of,
            "recency_days": 7,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "final": "",
        }
    )

    return out




if __name__ == "__main__":
    topic = "How Transformer Architecture Works (With Intuition, Math & Code)"
    run(topic)