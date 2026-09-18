import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# Fixed the space typo in "TAVILY_API_KEY"
tavily = TavilySearchResults(
    max_results=5,
    api_key=os.getenv("TAVILY_API_KEY")
)

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns Titles, URLs, Snippets"""
    # Fix: Use .invoke() instead of .search()
    results = tavily.invoke({"query": query})

    out = []
    for r in results:
        out.append(
            f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url')}\nSnippet: {r.get('content', '')[:300]}\n"
        )

    return "\n-------\n".join(out)



@tool
def scrape_url(url: str)-> str:
    "Scrape and return clean text content from a given  URL for deeper reading"
    try:
        resp=requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup=BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style","nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape URL:{str(e)}"


