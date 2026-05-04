from typing import List
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
load_dotenv()




def tavily_search(query: str, max_results: int) -> List[dict]:
    tool = TavilySearch(max_results=max_results)
    results = tool.invoke(query)

    normalized: List[dict] = []
    for r in results.get("results") or []:
        normalized.append({
            "title": r.get("title" , ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
            "published_at": r.get("published_at", "") or r.get("published_date", ""),
            "source": r.get("source", "")

        })

    return normalized