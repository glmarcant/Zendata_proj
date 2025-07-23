import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_openai import AzureOpenAIEmbeddings  # Si può sostituire con JinaEmbeddings successivamente
from langchain.vectorstores import Milvus
# from langchain.vectorstores import Quadrant  # Per estensione futura

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

class Embedder:
    """
    Modulo per convertire chunk di documenti in vettori e salvarli in un database vettoriale.
    Supporta Milvus e può essere esteso a Quadrant.
    """

    def __init__(self, embedder_type: str = "azure_openai", db_type: str = "milvus", **kwargs):
        """
        Args:
            embedder_type (str): Tipo di embedder ('azure_openai', 'jina').
            db_type (str): Tipo di database vettoriale ('milvus', 'quadrant').
            kwargs: Parametri aggiuntivi per embedder e database.
        """
        if embedder_type == "azure_openai":
            self.embedder = AzureOpenAIEmbeddings(
                api_key=api_key,
                azure_endpoint=endpoint,
                **kwargs
            )
        # elif embedder_type == "jina":
        #     from langchain.embeddings import JinaEmbeddings
        #     self.embedder = JinaEmbeddings(**kwargs)
        else:
            raise ValueError("Embedder non supportato.")

        self.db_type = db_type
        self.db = None  # Sarà inizializzato in save_embeddings

    def embed_chunks(self, chunks: List[Any]) -> List[Dict]:
        """
        Converte i chunk in vettori, mantenendo il nome del file di origine.
        Args:
            chunks (List[Any]): Lista di chunk (Document).
        Returns:
            List[Dict]: Lista di dict con 'vector', 'source', 'content'.
        """
        embeddings = self.embedder.embed_documents([chunk.page_content for chunk in chunks])
        results = []
        for chunk, vector in zip(chunks, embeddings):
            source = chunk.metadata.get("source", "unknown")
            results.append({
                "vector": vector,
                "source": source,
                "content": chunk.page_content
            })
        return results

    def save_embeddings(self, embedded_chunks: List[Dict], collection_name: str = "rag_chunks"):
        """
        Salva i vettori nel database vettoriale scelto.
        Args:
            embedded_chunks (List[Dict]): Lista di dict con vettori e metadati.
            collection_name (str): Nome della collezione nel database.
        """
        if self.db_type == "milvus":
            self.db = Milvus.from_embeddings(
                embeddings=[item["vector"] for item in embedded_chunks],
                metadatas=[{"source": item["source"]} for item in embedded_chunks],
                texts=[item["content"] for item in embedded_chunks],
                collection_name=collection_name
            )
        elif self.db_type == "quadrant":
            # Da implementare: Quadrant vectorstore
            pass
        else:
            raise ValueError("Database vettoriale non supportato.")

    def process(self, chunks: List[Any], collection_name: str = "rag_chunks"):
        """
        Pipeline completa: embedding + salvataggio.
        Args:
            chunks (List[Any]): Lista di chunk.
            collection_name (str): Nome della collezione.
        """
        embedded_chunks = self.embed_chunks(chunks)
        self.save_embeddings(embedded_chunks, collection_name=collection_name)



# nel sistema, invece di usare direttamente Milvus o Quadrant, usi il wrapper generico:

# vector_store = MilvusVectorStore()  # o QuadrantVectorStore()
# vector_store.add_documents(chunks)
# results = vector_store.search("qual è la capitale d’Italia?", top_k=5)