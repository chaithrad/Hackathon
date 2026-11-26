# ContractGuard AI — Vendor Contract Redline & Playbook Reviewer (Problem 3B)

## Problem Statement
**ContractGuardAI: Vendor Contract Redline & Playbook Reviewer**
**Summary**: Cut legal review turnaround time by automatically scoring vendor agreements against internal negotiation playbooks and suggesting compliant redlines.
**Problem** **Description**: Legal and procurement teams spend hours manually reviewing routine vendor NDAS and MSAs for risky liability caps, IP traps, or unfavorable governing law clauses. Participants must build a RAG-powered contract reviewer that evaluates agreements against an internal Legal Playbook and generates safe redline wording.
**Expected** **Functionalities**:
1. Semantic retriever indexing an internal company Legal Negotiation Playbook.
2. Clause-by-clause contract parser evaluating uploaded NDAS/MSAs against retrieved playbook rules.
3. Risk scorecard (Low / Medium / High) with plain-English explanations of commercial exposure.
4. Drop-in redline text generator bringing non-compliant clauses into company policy alignment.

**Suggested** **Data** **Requirements**: 1 internal Legal Negotiation Playbook
(Markdown/PDF) and 3 sample vendor contracts (NDAS/MSAs) containing intentional high-risk clauses.


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
