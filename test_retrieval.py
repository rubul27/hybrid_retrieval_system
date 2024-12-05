import qdrant_client
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch

class HybridRetrieval:
    def __init__(self):
        embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
        
        # Initialize dense embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        
        # Initialize Qdrant client for dense vector search
        self.qdrant_client = qdrant_client.QdrantClient(url="http://localhost:6333")
        
        # Initialize TF-IDF vectorizer for sparse retrieval
        self.tfidf_vectorizer = TfidfVectorizer()

        # Fit the TF-IDF vectorizer on your corpus if needed (this should be done during ingestion)
        # self.tfidf_vectorizer.fit(corpus)  # Fit on the full text corpus

    def generate_embeddings(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
        return embeddings

    def dense_search(self, query, top_k=5):
        query_embeddings = self.generate_embeddings([query])
        # Ensure the embedding is a flat list of floats
        query_vector = query_embeddings[0]

        search_result = self.qdrant_client.search(
            collection_name='{dense_collection}',  # Replace with actual collection name
            query_vector=query_vector,  # Pass the correct format
            limit=top_k
        )
        
        results = [(result.payload, result.score) for result in search_result]
        return results

    
    def generate_tfidf_embeddings(self, texts):
        return self.tfidf_vectorizer.fit_transform([texts]).toarray()
    
    def pad_vector(self,vector, target_size):
        """Pads the vector with zeros to match the target size."""
        # Ensure the vector is a list
        vector = vector.tolist() if isinstance(vector, np.ndarray) else list(vector)
        
        # Calculate the number of zeros needed
        padding_size = target_size - len(vector)
        
        if padding_size > 0:
            return vector + [0.0] * padding_size
        return vector

    def sparse_search(self, query, top_k=5):
        # Generate TF-IDF vector for the query
        query_vector = self.generate_tfidf_embeddings(query)[0]

        # Pad the query vector to the desired dimension (4132 in this case)
        query_vector = self.pad_vector(query_vector, 4132)

        # Ensure the vector is converted to a format expected by Qdrant
        query_vector = query_vector if isinstance(query_vector, list) else query_vector.tolist()  
        
        search_results = self.qdrant_client.search(
            collection_name='sparse_collection',  # Replace with actual collection name
            query_vector=query_vector,  # Pass the correctly formatted vector
            limit=top_k
        )

        results = [(result.payload, result.score) for result in search_results]
        return results
    
    def hybrid_search(self, query, top_k=5):
        dense_results = self.dense_search(query, top_k=top_k)
        sparse_results = self.sparse_search(query, top_k=top_k)
        
        combined_results = dense_results + sparse_results
        combined_results = sorted(combined_results, key=lambda x: x[1], reverse=True)[:top_k]
        
        return combined_results

if __name__ == "__main__":
    hybrid_retrieval = HybridRetrieval()
    
    query = "Which major subsidiaries does Alphabet own?"
    
    results = hybrid_retrieval.hybrid_search(query, top_k=5)
    
    for result in results:
        print(f"Document: {result[0]['source']}, Score: {result[1]}")
        print(f"Text: {result[0]['text'][:200]}...")
        print("-" * 80)




