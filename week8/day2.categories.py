
import chromadb
# PERSIST_DIR = "chroma_store"
# COLLECTION_NAME = "products_vectorstore_backup"
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

DB = "products_vectorstore"
client = chromadb.PersistentClient(path=DB)
collection = client.get_or_create_collection('products')
MAXIMUM_DATAPOINTS = 30_000
CATEGORIES = [
    'Appliances',
    'Automotive',
    'Cell_Phones_and_Accessories',
    'Electronics',
    'Musical_Instruments',
    'Office_Products',
    'Tools_and_Home_Improvement',
    # 'Toys_and_Games',
    'Software',
    'Health_and_Personal_Care'
]

result = collection.get(include=['embeddings', 'documents', 'metadatas'], limit=MAXIMUM_DATAPOINTS)
print(len(result))

unique_categories = list({item['category'] for item in result['metadatas']})
print(unique_categories)

missing = set(CATEGORIES) - set(unique_categories)

if not missing:
    print("✅ 完整匹配")
else:
    print("⚠️ 缺少：", list(missing))