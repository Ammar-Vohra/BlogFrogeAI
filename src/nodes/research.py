from langchain_core.messages import HumanMessage, SystemMessage
from src.models.schemas import State, EvidencePack
from src.prompts.prompt import RESEARCH_SYSTEM
from typing import List
from src.services.llm_service import llm
from src.services.tavily_service import tavily_search
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger("research")


def research_node(state: State) -> dict:
    logger.info("Research node started")

    try:
        queries = state.get("queries", [])
        max_results = 6

        logger.info(f"Total queries received: {len(queries)}")

        raw_results: List[dict] = []

        for q in queries:
            logger.debug(f"Searching query: {q}")

            try:
                results = tavily_search(q, max_results=max_results)
                logger.debug(f"Results fetched: {len(results)} for query: {q}")

                raw_results.extend(results)

            except Exception as e:
                logger.error(f"Tavily search failed for query '{q}': {str(e)}")

        logger.info(f"Total raw results collected: {len(raw_results)}")

        if not raw_results:
            logger.warning("No research results found")
            return {"evidence": []}

        output = llm.with_structured_output(EvidencePack)

        logger.info("Extracting structured evidence from raw results")

        pack = output.invoke(
            [
                SystemMessage(content=RESEARCH_SYSTEM),
                HumanMessage(content=f"Raw Results: {raw_results}")
            ]
        )

        logger.info(f"Total evidence items before dedup: {len(pack.evidence)}")

        # Deduplicate by URL
        dedup = {}
        for e in pack.evidence:
            if e.url:
                dedup[e.url] = e

        final_evidence = list(dedup.values())

        logger.info(f"Total evidence after deduplication: {len(final_evidence)}")

        return {"evidence": final_evidence}

    except Exception as e:
        logger.error(f"Research node failed: {str(e)}")
        raise