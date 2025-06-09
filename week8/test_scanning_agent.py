from agents.scanner_agent import ScannerAgent
agent = ScannerAgent(show_progress=True)
print("Starting scan, please wait...")
result = agent.scan()
print(f"Top Deals found: {len(result.deals)}")
print(f"Top 5 Deals: {result.deals}")
