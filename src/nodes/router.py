from langchain_core.messages import HumanMessage, SystemMessage
from src.models.schemas import State, RouterDecision
from src.prompts.prompt import ROUTER_SYSTEM
from src. services.llm_service import llm
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("router")


def router_node(state: State) -> dict:
    logger.info("Router started")

    try:
        topic = state.get("topic", "")
        logger.info(f"Routing topic: {topic}")

        decision = llm.with_structured_output(RouterDecision)

        decider = decision.invoke(
            [
                SystemMessage(content=ROUTER_SYSTEM),
                HumanMessage(content=f"{topic}")
            ]
        )

        logger.info(
            f"Routing decision → mode: {decider.mode}, needs_research: {decider.needs_research}"
        )

        if decider.queries:
            logger.debug(f"Generated queries: {decider.queries}")

        return {
            "needs_research": decider.needs_research,
            "mode": decider.mode,
            "queries": decider.queries,
        }

    except Exception as e:
        logger.error(f"Router failed: {str(e)}")
        raise


def route_next(state: State) -> str:
    next_step = "research" if state.get("needs_research", False) else "orchestrator"

    logger.info(f"Routing to next node: {next_step}")

    return next_step