from dotenv import load_dotenv
import chromadb
from agents.planning_agent import PlanningAgent

load_dotenv(override=True)
DB = "products_vectorstore"

client = chromadb.PersistentClient(path=DB)
collection = client.get_or_create_collection('products')
print (f"Get collection '{collection.name}'. ")
planner = PlanningAgent(collection)
print ("Created planningAgent, please wait...")
result = planner.plan()
print (f"Planning completed with result: {result}. ")