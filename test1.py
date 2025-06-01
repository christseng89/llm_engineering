import chromadb

client = chromadb.PersistentClient(path="products_vectorstore")
print(client.list_collections())