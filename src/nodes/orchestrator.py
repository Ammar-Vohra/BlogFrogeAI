from langchain_core.messages import HumanMessage, SystemMessage
from src.models.schemas import State, Plan
from src.prompts.prompt import ORCH_SYSTEM
from typing import List
from src.services.llm_service import llm
from langgraph.types import Send
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("orchestrator")


def orchestrator_node(state: State) -> dict:
    logger.info("Orchestrator started")

    try:
        planner = llm.with_structured_output(Plan)

        evidence = state.get("evidence", [])
        mode = state.get("mode", "closed_book")

        logger.debug(f"Mode: {mode}")
        logger.debug(f"Number of evidence items: {len(evidence)}")

        plan = planner.invoke(
            [
                SystemMessage(content=ORCH_SYSTEM),
                HumanMessage(
                    content=(
                        f"Topic: {state['topic']}\n"
                        f"Mode: {mode}\n"
                        f"Evidence: {evidence}\n"
                        f"{[e.model_dump() for e in evidence][:16]}"
                    )
                )
            ]
        )

        logger.info(f"Plan generated with {len(plan.task)} sections")

        return {"plan": plan}

    except Exception as e:
        logger.error(f"Orchestrator failed: {str(e)}")
        raise


def fanout(state: State):
    logger.info("Fanout started")

    sends = []

    try:
        tasks = state["plan"].task
        logger.info(f"Total tasks to dispatch: {len(tasks)}")

        for task in tasks:
            logger.debug(f"Dispatching task ID: {task.id}, Title: {task.title}")

            sends.append(
                Send(
                    "worker",
                    {
                        "task": task.model_dump(),
                        "topic": state["topic"],
                        "mode": state["mode"],
                        "plan": state["plan"].model_dump(),
                        "evidence": [e.model_dump() for e in state.get("evidence", [])],
                    },
                )
            )

        logger.info("Fanout completed")

        return sends

    except Exception as e:
        logger.error(f"Fanout failed: {str(e)}")
        raise