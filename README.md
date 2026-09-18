# 🔎 Multi-Agent Research System

A Streamlit-based application that runs a 4-step multi-agent research pipeline:
**Search → Read → Write → Critique**, all in one workflow.

Powered by [LangChain](https://www.langchain.com/), [Tavily](https://tavily.com/), and your choice of LLM provider: [Google Gemini](https://aistudio.google.com/) (recommended free tier), [Groq](https://console.groq.com/), or [Mistral AI](https://mistral.ai/).

---

## ✨ Features

- 🔍 **Search Agent** – finds recent, reliable sources on a given topic
- 📖 **Reader Agent** – scrapes the most relevant URL for deeper content
- ✍️ **Writer Chain** – drafts a full structured research report
- 🧐 **Critic Chain** – reviews the report and provides scored feedback
- 📄 Download the final report as a `.md` file
- 🖥️ Clean Streamlit UI with progress tracking and tabs

---

## 🧠 How the Pipeline Works

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Search     │───▶│  Reader     │───▶│  Writer     │───▶│  Critic     │
│  Agent      │    │  Agent      │    │  Chain      │    │  Chain      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
    Web search        URL scraping      Structured report   Scored review
```

### Step 1 — Search Agent (`build_search_agent`)
Uses the **Tavily** web search tool to gather recent, reliable sources for the user's topic. Results include titles, URLs, and snippets.

### Step 2 — Reader Agent (`build_reader_agent`)
Receives the search results, picks the most relevant URL, and **scrapes it** for detailed content using `BeautifulSoup`.

### Step 3 — Writer Chain (`writer_chain`)
Combines the search results and scraped content, then asks Mistral to write a detailed report structured as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (only URLs found in the research context)

### Step 4 — Critic Chain (`critic_chain`)
Reviews the draft and returns a structured evaluation:
- Score (x/10)
- Strengths
- Areas to Improve
- One Line Verdict

---

## 📁 Project Structure

```
.
├── app.py          # Streamlit UI (entry point)
├── agents.py       # Search/Reader agents + Writer/Critic chains
├── pipeline.py     # Programmatic pipeline runner
├── tools.py        # Custom LangChain tools (web search, URL scraping)
├── requirements.txt# Python dependencies
└── Procfile        # Railway/Heroku start command
```

---

## 🚀 Running Locally

1. **Clone the repo**

   ```bash
   git clone https://github.com/Azansoh/Multi-Agent-DeepResearch.git
   cd Multi-Agent-DeepResearch
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS / Linux
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root. Provide **one** LLM key and the Tavily key:

   ```env
   # Recommended: Google Gemini (generous free tier)
   GOOGLE_API_KEY=your_google_ai_studio_key
   GEMINI_MODEL=gemini-3.6-flash

   # Or Groq
   # GROQ_API_KEY=your_groq_key
   # GROQ_MODEL=openai/gpt-oss-120b

   # Or Mistral (fallback; free tier is heavily rate-limited)
   # MISTRAL_API_KEY=your_mistral_key

   TAVILY_API_KEY=your_tavily_api_key
   ```

   The provider is auto-detected from whichever key is present (Gemini → Groq → Mistral).
   Set `LLM_PROVIDER=google|groq|mistral` to force one.

   > Get keys from [Google AI Studio](https://aistudio.google.com/apikey), [Groq Console](https://console.groq.com/keys), [Mistral AI Console](https://console.mistral.ai/), and [Tavily](https://tavily.com/).

5. **Run the app**

   ```bash
   streamlit run app.py
   ```

   Open the URL shown in the terminal (default: `http://localhost:8501`).

### Optional: Run the pipeline without the UI

```bash
python pipeline.py
```

---

## ☁️ Deploying on Streamlit Community Cloud

1. Push this repo to GitHub.
2. In [share.streamlit.io](https://share.streamlit.io/), click **Create app** → **Deploy from GitHub**.
3. Set **Main file path** to `app.py`.
4. Open **Advanced settings → Secrets** and add:

   ```toml
   GOOGLE_API_KEY = "your_google_ai_studio_key"
   TAVILY_API_KEY = "your_tavily_api_key"
   ```

   > `.env` is git-ignored, so it is **not** deployed. Streamlit Cloud only reads
   > secrets from this dashboard (they are exposed as environment variables).
5. Click **Deploy**. After changing a secret, use **Manage app → Reboot** so the new value takes effect.

---

## ☁️ Deploying on Railway

1. Push this repo to GitHub.
2. In [Railway](https://railway.app/), create a **New Project** → **Deploy from GitHub repo**.
3. Add the environment variables in your service **Variables** tab:
   - `GOOGLE_API_KEY` (or `GROQ_API_KEY` / `MISTRAL_API_KEY`)
   - `TAVILY_API_KEY`
4. Railway reads the `Procfile` and starts the app:

   ```
   web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```

5. Railway auto-detects `requirements.txt` and installs dependencies, then serves the app on the assigned port.

---

## 🔐 Required API Keys

| Variable                          | Provider             | Purpose                               |
| --------------------------------- | -------------------- | ------------------------------------- |
| `GOOGLE_API_KEY` / `GROQ_API_KEY` / `MISTRAL_API_KEY` | Gemini (recommended) / Groq / Mistral | LLM for search, writing, and critique |
| `TAVILY_API_KEY`                  | Tavily               | Web search tool                       |

---

## 📦 Dependencies

- LangChain (`langchain`, `langchain-core`, `langchain-community`)
- LLM providers (`langchain-google-genai`, `langchain-groq`, `langchain-mistralai`, `mistralai`)
- Tavily (`tavily-python`)
- Web scraping (`beautifulsoup4`, `lxml`)
- HTTP (`requests`, `httpx`)
- Others (`python-dotenv`, `pydantic`, `rich`, `streamlit`)
