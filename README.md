---
title: DataPulse AI
emoji: 📊
sdk: streamlit
app_file: app.py
python_version: "3.11"
---

# DataPulse AI
Ask questions across CSV/Parquet files and PDF reports in one chat.
Text-to-SQL (DuckDB) with a self-healing retry loop, PDF RAG (Docling + LanceDB), LangGraph routing, cited answers, Plotly charts.

## Run locally
```
pip install -r requirements.txt
export LLM_API_KEY=your_key          # any OpenAI-compatible API (Groq, OpenRouter, Ollama)
export LLM_BASE_URL=https://api.groq.com/openai/v1
export LLM_MODEL=qwen/qwen3-32b      # check your provider's model list
streamlit run app.py
```
Demo login: `demo` / `DataPulse2026` (override with DEMO_USER / DEMO_PASS).
For self-hosting set `LLM_BASE_URL=http://localhost:11434/v1` and `LLM_MODEL=qwen2.5-coder:14b` (Ollama).
