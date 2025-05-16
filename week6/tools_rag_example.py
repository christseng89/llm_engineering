from langchain.agents import initialize_agent, Tool
from langchain.tools import tool
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI

# 1. Vector store (RAG)
embedding = HuggingFaceEmbeddings()
vectorstore = Chroma(persist_directory="db", embedding_function=embedding)
retriever = vectorstore.as_retriever()

rag_qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), retriever=retriever, chain_type="stuff"
)

# 2. Define a tool (e.g., tax refund calculator)
@tool
def calculate_refund(order_id: str, unit_price: float, quantity: int = 1) -> str:
    if quantity <= 0:
        return f"Invalid quantity for order {order_id}."
    if unit_price < 0:
        return f"Invalid price for order {order_id}."

    total_refund = unit_price * quantity
    return (
        f"Order {order_id}: Refund is ${total_refund:.2f} "
        f"for {quantity} item(s) at ${unit_price:.2f} each."
    )

# 3. Define tools list
tools = [
    Tool(name="RefundCalculator", func=calculate_refund, description="Calculate refund for an order"),
    Tool(name="PolicyRAG", func=rag_qa.run, description="Answer company policy questions")
]

# 4. Setup agent (LLM decides which tool to use)
agent = initialize_agent(
    tools=tools,
    llm=OpenAI(),
    agent="zero-shot-react-description",
    verbose=True
)

# 5. Ask question (agent chooses between RAG or Tool or both)
response = agent.run("What’s the refund policy for digital items? Also, calculate refund for order 12345.")
print(response)
