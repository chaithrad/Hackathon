import streamlit as st
import hashlib
import json
import os
import pandas as pd
from datetime import datetime
from pypdf import PdfReader
from dotenv import load_dotenv
from contract_graph import build_contract_graph

load_dotenv()

DECISIONS_FILE = "decisions_log.json"

def load_decisions() -> list:
    if not os.path.exists(DECISIONS_FILE):
        with open(DECISIONS_FILE, "w") as f:
            json.dump([], f)
    with open(DECISIONS_FILE, "r") as f:
        return json.load(f)

def save_decision(record: dict) -> bool:
    records = load_decisions()
    for r in records:
        if r["doc_hash"] == record["doc_hash"] and r["vendor_id"] == record["vendor_id"]:
            return False
    records.append(record)
    with open(DECISIONS_FILE, "w") as f:
        json.dump(records, f, indent=4)
    return True

def extract_text_from_upload(uploaded_file) -> str:
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    return uploaded_file.getvalue().decode("utf-8")

st.set_page_config(page_title="ContractGuard AI | Problem 3B", page_icon="⚖️", layout="wide")
st.title("⚖️ ContractGuard AI: Vendor Contract Redline & Playbook Reviewer")
st.caption("Problem 3B • Gemini 3.1 Flash Lite • FAISS Playbook RAG • MCP Vendor Server • LangGraph")
st.warning("🛡️ **Responsible AI Notice:** AI-assisted legal review tool. All redlines require final Human-in-the-Loop Legal/Procurement sign-off.")

with open("vendors.json", "r") as f:
    vendors_data = json.load(f)

with st.sidebar:
    st.header("🏢 Vendor Context (MCP)")
    selected_vid = st.selectbox(
        "Select Vendor ID:",
        options=list(vendors_data.keys()),
        format_func=lambda vid: f"{vid} — {vendors_data[vid]['vendor_name']}"
    )
    st.json(vendors_data[selected_vid])
    contract_val = st.number_input("Deal Value (USD):", min_value=10000, value=120000, step=10000)

    st.divider()
    st.subheader("📂 Sample Contracts (3B Data)")
    sample_files = sorted(os.listdir("samples")) if os.path.exists("samples") else []
    selected_sample = st.selectbox("Load a sample NDA/MSA:", ["None"] + sample_files)

uploaded_doc = st.file_uploader("Upload Vendor NDA / MSA (.pdf, .txt, .md)", type=["pdf", "txt", "md"])

contract_text = ""
doc_name = ""
if uploaded_doc:
    contract_text = extract_text_from_upload(uploaded_doc)
    doc_name = uploaded_doc.name
elif selected_sample != "None":
    doc_name = selected_sample
    with open(os.path.join("samples", selected_sample), "r", encoding="utf-8") as f:
        contract_text = f.read()

if contract_text:
    doc_hash = hashlib.md5(contract_text.encode("utf-8")).hexdigest()[:12]
    with st.expander(f"📄 Loaded Contract: {doc_name} (Hash: {doc_hash})", expanded=False):
        st.text(contract_text)

    if st.button("🚀 Run Playbook & Redline Review", type="primary"):
        try:
            graph = build_contract_graph()
            with st.spinner("Running LangGraph (MCP Vendor Check → FAISS Legal Playbook → Clause Redliner)..."):
                result_state = graph.invoke({
                    "vendor_id": selected_vid,
                    "contract_value_usd": float(contract_val),
                    "contract_text": contract_text,
                    "execution_log": []
                })
            st.session_state["last_review"] = {
                "state": result_state,
                "doc_hash": doc_hash,
                "doc_name": doc_name,
                "vendor_id": selected_vid
            }
        except Exception as e:
            st.error(f"Error running agent: {e}")

if "last_review" in st.session_state:
    rev = st.session_state["last_review"]
    state = rev["state"]
    report = state["final_report"]

    with st.expander("🔍 LangGraph + FastMCP + FAISS Execution Trace", expanded=True):
        for step in state["execution_log"]:
            st.write(step)
        st.code(state["mcp_vendor_context"])

    rating = report["overall_risk_rating"]
    if rating == "High":
        st.error(f"🚨 Overall Risk Scorecard: HIGH ({report['risk_score']}/100)")
    elif rating == "Medium":
        st.warning(f"⚠️ Overall Risk Scorecard: MEDIUM ({report['risk_score']}/100)")
    else:
        st.success(f"✅ Overall Risk Scorecard: LOW ({report['risk_score']}/100)")

    st.write(f"**Executive Summary:** {report['executive_summary']}")
    st.caption(f"🛡️ **Responsible AI Note:** {report['responsible_ai_governance_note']}")

    st.subheader("📑 Clause-by-Clause Scorecard & Drop-In Redlines")
    for idx, clause in enumerate(report["clauses"], 1):
        with st.container(border=True):
            c_title = f"**{idx}. {clause['clause_title']}** — Risk: `{clause['risk_level']}`"
            if clause["risk_level"] == "High":
                st.error(c_title)
            elif clause["risk_level"] == "Medium":
                st.warning(c_title)
            else:
                st.success(c_title)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Original Vendor Wording:**")
                st.code(clause["contract_excerpt"], language="text")
                st.markdown(f"**Commercial Exposure:** {clause['plain_english_exposure']}")
            with col_b:
                st.markdown("**✅ Drop-In Compliant Redline:**")
                st.code(clause["compliant_redline_text"], language="text")

    st.subheader("👩‍⚖️ Legal Sign-Off (Persists Across Page Reloads)")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("✅ Accept Contract"):
            ok = save_decision({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "doc_hash": rev["doc_hash"],
                "document": rev["doc_name"],
                "vendor_id": rev["vendor_id"],
                "risk_rating": rating,
                "risk_score": report["risk_score"],
                "decision": "ACCEPTED"
            })
            st.success("Decision saved! Reload page to verify.") if ok else st.warning("Duplicate import blocked (0 rows added).")
    with b2:
        if st.button("🛑 Hold & Send Redlines to Vendor"):
            ok = save_decision({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "doc_hash": rev["doc_hash"],
                "document": rev["doc_name"],
                "vendor_id": rev["vendor_id"],
                "risk_rating": rating,
                "risk_score": report["risk_score"],
                "decision": "HOLD_REDLINE_SENT"
            })
            st.error("Hold recorded! Reload page to verify.") if ok else st.warning("Duplicate import blocked (0 rows added).")

st.divider()
st.subheader("📋 Reconciled Legal Review Register (Persists After Reload)")
records = load_decisions()
if records:
    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        label="📥 Export Reconciled Register (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="contract_redline_register.csv",
        mime="text/csv"
    )
else:
    st.info("No contract decisions recorded yet. Run a review above and click Accept or Hold.")
