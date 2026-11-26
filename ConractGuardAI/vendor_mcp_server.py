from mcp.server.mcpserver import MCPServer
import json

mcp = MCPServer("ContractGuard-Vendor-MCP")

@mcp.tool()
def get_vendor_profile(vendor_id: str) -> str:
    """Fetch vendor tier, approved budget cap, and dispute history from corporate registry."""
    with open("vendors.json", "r") as f:
        vendors = json.load(f)
    vendor = vendors.get(vendor_id.upper())
    if not vendor:
        return f"WARNING: Vendor ID {vendor_id} not found in approved vendor master."
    return json.dumps(vendor, indent=2)

@mcp.tool()
def verify_liability_floor(proposed_cap_usd: float, contract_value_usd: float) -> str:
    """Deterministic check to ensure vendor liability cap meets corporate minimum ($100,000 or 12-month value)."""
    required_floor = max(100000.0, contract_value_usd)
    if proposed_cap_usd < required_floor:
        return f"POLICY BREACH: Proposed cap ${proposed_cap_usd:,.2f} is below required floor of ${required_floor:,.2f}."
    return f"COMPLIANT: Proposed cap ${proposed_cap_usd:,.2f} meets required floor."

if __name__ == "__main__":
    mcp.run(transport="stdio")
