from dotenv import load_dotenv
import chromadb
from agents.planning_agent import PlanningAgent

load_dotenv(override=True)
DB = "products_vectorstore"

client = chromadb.PersistentClient(path=DB)
collection = client.get_or_create_collection('products')
print (f"Get Collection '{collection.name}'. ")
planner = PlanningAgent(collection)
print ("Created PlanningAgent. ")
planner.plan()
