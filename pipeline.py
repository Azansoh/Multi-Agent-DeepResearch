import re

from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain

_URL_RE = re.compile(r"https?://[^\s)\]}>'\"]+")
_REFUSAL_PHRASES = [
    "provide the full link", "please provide the", "could you please provide",
    "could you provide", "please share the", "doesn't include an actual url",
    "no url was provided", "no url", "unable to locate", "can't perform",
    "cannot perform", "i'd be happy to", "cannot scrape", "unable to scrape",
    "could not scrape", "no content",
]


def _extract_urls(text: str):
    if not text:
        return []
    urls = [u.rstrip(".,;:!?") for u in _URL_RE.findall(text)]
    return list(dict.fromkeys(u for u in urls if u.startswith("http")))


def _looks_scraped(content: str) -> bool:
    content = content or ""
    if len(content) < 200:
        return False
    lowered = content.lower()
    return not any(phrase in lowered for phrase in _REFUSAL_PHRASES)


def run_research_pipeline(topic: str) -> dict:
    state = {}

    # Step 1 - Search Agent
    print("\n" + "=" * 50)
    print("Step 1 - Search agent is working...")
    print("=" * 50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user",
            f"Find recent, reliable and detailed information about: {topic}. "
            f"In your final answer, include the key facts AND every source URL "
            f"you found, one per line, each prefixed with 'URL:'. Do not omit any URL.")]
    })

    # Fixed key to 'search_results' (plural)
    state["search_results"] = str(search_result['messages'][-1].content)
    print("\nSearch Result:\n", state["search_results"])

    # Step 2 - Reader Agent
    print("\n" + "=" * 50)
    print("Step 2 - Reader is scraping the best source...")
    print("=" * 50)

    from tools import scrape_url as scrape_url_tool

    all_messages_text = "\n".join(
        str(m.content) for m in search_result["messages"] if m.content
    )
    candidate_urls = _extract_urls(all_messages_text)

    scraped_content = ""
    for url in candidate_urls[:5]:
        try:
            result = str(scrape_url_tool.invoke({"url": url}))
        except Exception:
            continue
        if _looks_scraped(result):
            scraped_content = result
            break

    if not _looks_scraped(scraped_content):
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke({
            "messages": [(
                "user",
                f"Scrape detailed content from a real URL in the search results below.\n\n"
                f"STRICT RULES:\n"
                f"1. You MUST call the scrape_url tool with a real URL from the list. "
                f"Never reply in plain text and never ask the user for a URL.\n"
                f"2. Prefer the most relevant URL; otherwise use the first valid 'URL:' entry.\n"
                f"3. Report back the scraped text returned by the tool.\n\n"
                f"Search Results about '{topic}':\n{state['search_results'][:4000]}"
            )]
        })
        scraped_content = str(reader_result['messages'][-1].content)

    state['scraped_content'] = scraped_content
    print("\nScraped Content:\n", state['scraped_content'])

    # Step 3 - Writer Chain
    print("\n" + "=" * 50)
    print("Step 3 - Writer is drafting the report...")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    print("\nFinal Report:\n", state['report'])

    # Step 4 - Critic Chain
    print("\n" + "=" * 50)
    print("Step 4 - Critic is reviewing the report...")
    print("=" * 50)

    state['feedback'] = critic_chain.invoke({
        "report": state['report']
    })

    print("\nCritic Report:\n", state['feedback'])

    return state


if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    run_research_pipeline(topic)