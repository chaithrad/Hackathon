import json
from dotenv import load_dotenv
from vendor_mcp_server import get_vendor_profile, verify_liability_floor
from playbook_rag import retrieve_playbook_rules

load_dotenv()

def test_vendor_profile_lookup():
    data = json.loads(get_vendor_profile("VND-102"))
    assert data["vendor_name"] == "Apex Offshore Data Corp"
    assert data["past_disputes"] == 2

def test_liability_floor_check():
    breach = verify_liability_floor(20000.0, 120000.0)
    assert "POLICY BREACH" in breach
    ok = verify_liability_floor(150000.0, 120000.0)
    assert "COMPLIANT" in ok

def test_playbook_semantic_retrieval():
    rules = retrieve_playbook_rules("Cayman Islands governing law and London arbitration")
    assert "Rule 3" in rules
