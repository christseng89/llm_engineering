from dotenv import load_dotenv
import chromadb
from agents.planning_agent_async import PlanningAgentAsync

load_dotenv(override=True)
DB = "products_vectorstore"

client = chromadb.PersistentClient(path=DB)
collection = client.get_or_create_collection('products')
print (f"Get Collection '{collection.name}'. ")
planner = PlanningAgentAsync(collection)
print ("Created PlanningAgentAsync. ")
planner.plan()
