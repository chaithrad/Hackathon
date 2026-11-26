# ContractGuard AI — Vendor Contract Redline & Playbook Reviewer (Problem 3B)

## Architecture
- **FastMCP Server (`vendor_mcp_server.py`)**: Fetches vendor tier, approved budget, and deterministic liability floors.
- **Semantic Retriever (`playbook_rag.py`)**: Indexes `legal_playbook.md` into FAISS using HuggingFace `all-MiniLM-L6-v2` embeddings.
- **LangGraph Workflow (`contract_graph.py`)**: Stateful 3-node multi-agent pipeline (`mcp_vendor` -> `rag_playbook` -> `redline_agent`) powered by `gemini-3.8-flash`.
- **Streamlit UI (`app.py`)**: Side-by-side clause redline comparison, duplicate import guard, persistent Human-in-the-Loop decisions across page reloads, and CSV export.

## Quick Start
1. `pip install -r requirements.txt`
2. Add `GOOGLE_API_KEY` in `.env`
3. Run tests: `pytest`
4. Launch app: `streamlit run app.py`
