from qdrant_client import QdrantClient

# Initialize the Qdrant client
client = QdrantClient(host='localhost', port=6333)  # Adjust host and port as necessary

# Specify the name of the collection you want to delete
collection_name = 'sparse_collection'

# Delete the collection
client.delete_collection(collection_name=collection_name)

print(f"Collection  has been deleted.") 