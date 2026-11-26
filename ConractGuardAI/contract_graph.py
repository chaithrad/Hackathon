import asyncio
from typing import TypedDict, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from vendor_mcp_server import mcp as vendor_mcp_server
from playbook_rag import retrieve_playbook_rules

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

class ClauseEvaluation(BaseModel):
    clause_title: str
    contract_excerpt: str
    risk_level: str = Field(description="Low, Medium, or High")
    plain_english_exposure: str = Field(description="Clear business/legal risk explanation")
    compliant_redline_text: str = Field(description="Drop-in replacement wording aligned with the Legal Playbook")

class ContractReviewReport(BaseModel):
    overall_risk_rating: str = Field(description="Low, Medium, or High")
    risk_score: int = Field(description="0 to 100 numeric risk score")
    executive_summary: str
    clauses: List[ClauseEvaluation]
    responsible_ai_governance_note: str = Field(description="Transparency & human legal counsel sign-off note")

class ContractGraphState(TypedDict):
    vendor_id: str
    contract_value_usd: float
    contract_text: str
    mcp_vendor_context: str
    retrieved_playbook: str
    final_report: dict
    execution_log: List[str]

async def _run_mcp_checks(vendor_id: str, contract_value: float):
    v_res = await vendor_mcp_server.call_tool("get_vendor_profile", {"vendor_id": vendor_id})
    cap_res = await vendor_mcp_server.call_tool(
        "verify_liability_floor",
        {"proposed_cap_usd": 50000.0, "contract_value_usd": contract_value}
    )
    v_text = v_res.content[0].text if getattr(v_res, "content", None) else str(v_res)
    cap_text = cap_res.content[0].text if getattr(cap_res, "content", None) else str(cap_res)
    return f"Vendor Profile:\n{v_text}\n\nLiability Floor Check:\n{cap_text}"

def mcp_vendor_node(state: ContractGraphState):
    mcp_out = asyncio.run(_run_mcp_checks(state["vendor_id"], state["contract_value_usd"]))
    log = state.get("execution_log", []) + ["🔌 MCP Node: Fetched vendor risk tier & verified liability floors via MCP Server."]
    return {"mcp_vendor_context": mcp_out, "execution_log": log}

def rag_playbook_node(state: ContractGraphState):
    rules = retrieve_playbook_rules(state["contract_text"])
    log = state["execution_log"] + ["📚 RAG Node: Retrieved matching rules from legal_playbook.md via FAISS."]
    return {"retrieved_playbook": rules, "execution_log": log}

def clause_redline_node(state: ContractGraphState):
    structured_llm = llm.with_structured_output(ContractReviewReport)
    prompt = f"""
    You are an enterprise Legal Contract Reviewer. Perform a clause-by-clause review of the vendor agreement below.
    Compare every clause strictly against the retrieved Legal Negotiation Playbook and Vendor MCP Context.
    Provide a Low/Medium/High risk rating, plain-English exposure, and exact drop-in redline replacement text.

    1. Contract Text:
    {state['contract_text']}

    2. Vendor & Budget Context (from MCP Server):
    {state['mcp_vendor_context']}

    3. Retrieved Legal Playbook Rules (from FAISS RAG):
    {state['retrieved_playbook']}
    """
    report = structured_llm.invoke(prompt)
    log = state["execution_log"] + [f"⚖️ Redline Agent Node: Evaluated {len(report.clauses)} clauses (Overall Risk: {report.overall_risk_rating})."]
    return {"final_report": report.model_dump(), "execution_log": log}

def build_contract_graph():
    workflow = StateGraph(ContractGraphState)
    workflow.add_node("mcp_vendor", mcp_vendor_node)
    workflow.add_node("rag_playbook", rag_playbook_node)
    workflow.add_node("redline_agent", clause_redline_node)

    workflow.set_entry_point("mcp_vendor")
    workflow.add_edge("mcp_vendor", "rag_playbook")
    workflow.add_edge("rag_playbook", "redline_agent")
    workflow.add_edge("redline_agent", END)

    return workflow.compile()
