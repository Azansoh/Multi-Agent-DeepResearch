import os
import random

import httpx
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception, stop_after_attempt

# LangChain Agent factory function
from langchain.agents import create_agent

# Mistral AI Chat model integration
from langchain_mistralai import ChatMistralAI

# Prompt template helper to construct chat message lists
from langchain_core.prompts import ChatPromptTemplate

# Output parser to convert LLM output objects directly to raw strings
from langchain_core.output_parsers import StrOutputParser

# Custom local tools for web searching and URL scraping
from tools import web_search, scrape_url

# Load environment variables (e.g., MISTRAL_API_KEY, TAVILY_API_KEY) from .env file
load_dotenv()


def _is_retryable_error(exc: BaseException) -> bool:
    """Retry only on rate limits (429) and server errors (5xx)."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or 500 <= status <= 599
    return isinstance(exc, (httpx.RequestError, httpx.StreamError))


def _retry_wait(retry_state) -> float:
    """Honor Mistral's Retry-After header, otherwise use exponential backoff."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, httpx.HTTPStatusError):
        retry_after = exc.response.headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
    return min(2 ** (retry_state.attempt_number - 1), 60.0) + random.uniform(0, 1)


class ResilientChatMistralAI(ChatMistralAI):
    """ChatMistralAI that also retries HTTP 429 / 5xx errors with backoff.

    langchain-mistralai only retries network/stream errors, so a single 429
    from Mistral otherwise aborts the whole pipeline.
    """

    def completion_with_retry(self, run_manager=None, **kwargs):
        @retry(
            reraise=True,
            retry=retry_if_exception(_is_retryable_error),
            wait=_retry_wait,
            stop=stop_after_attempt(self.max_retries + 1),
        )
        def _call():
            return super(ResilientChatMistralAI, self).completion_with_retry(
                run_manager=run_manager, **kwargs
            )

        return _call()


# Initialize the primary Large Language Model (Mistral Small)
llm = ResilientChatMistralAI(
    model="mistral-small-2506",
    max_retries=6,
)


# Function to create an Agent equipped with the web search tool
def build_search_agent():
    """Builds an agent capable of executing web searches to gather information."""
    return create_agent(
        model=llm,
        tools=[web_search]
    )


# Function to create an Agent equipped with the URL scraping tool
def build_reader_agent():
    """Builds an agent capable of reading/scraping detailed text from specific web links."""
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )


# ==========================================
# WRITER CHAIN
# ==========================================

# Define the prompt structure for the Report Writer
# Updated Writer Prompt snippet in agents.py
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. Write clear, structured, and factual reports. "
        "STRICT RULES:\n"
        "1. Do NOT include conversational closing questions (e.g., 'Would you like to know more?').\n"
        "2. Only cite URLs explicitly provided in the research context. Never invent URLs."
    ),
    (
        "human",
        """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list ONLY URLs found directly in the research context, or state 'No direct URLs available')
"""
    )
])

# Combine Prompt, Model, and Output Parser into a runnable LCEL chain
# Input: {"topic": "...", "research": "..."} -> Output: Raw string report
writer_chain = writer_prompt | llm | StrOutputParser()


# ==========================================
# CRITIC CHAIN
# ==========================================

# Define the prompt structure for the Report Critic
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sharp and constructive research critic. Be honest and specific."
    ),
    (
        "human",
        """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in the exact format:

Score: x/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One Line Verdict:
...
"""
    )
])

# Combine Prompt, Model, and Output Parser into a runnable LCEL chain
# Input: {"report": "..."} -> Output: Formatted review string
critic_chain = critic_prompt | llm | StrOutputParser()


# ==========================================
# EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # Example topic to test the full pipeline
    target_topic = "Impact of Quantum Computing on Modern Cryptography"

    # 1. Instantiate the Search Agent and query for information
    search_agent = build_search_agent()
    agent_response = search_agent.invoke({
        "messages": [("user", f"Search and gather recent facts on: {target_topic}")]
    })
    
    # Extract the final answer content from the agent's message list
    research_data = agent_response["messages"][-1].content

    # 2. Pass research to Writer Chain to construct the report draft
    draft_report = writer_chain.invoke({
        "topic": target_topic,
        "research": research_data
    })

    # 3. Pass draft report to Critic Chain for feedback and evaluation
    critic_evaluation = critic_chain.invoke({
        "report": draft_report
    })

    # Print results
    print("=== DRAFT REPORT ===\n", draft_report)
    print("\n=== CRITIC FEEDBACK ===\n", critic_evaluation)