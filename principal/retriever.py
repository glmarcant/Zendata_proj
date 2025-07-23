from langchain.vectorstores import Milvus
# from langchain.vectorstores import Quadrant  # Per estensione futura
from langchain.schema import Document
from typing import List, Any

class Retriever:
    """
    Modulo per recuperare i chunk più rilevanti da un database vettoriale.
    Supporta Milvus e può essere esteso a Quadrant.
    """

    def __init__(self, db_type: str = "milvus", collection_name: str = "rag_chunks", **kwargs):
        """
        Args:
            db_type (str): Tipo di database vettoriale ('milvus', 'quadrant').
            collection_name (str): Nome della collezione.
            kwargs: Parametri aggiuntivi per la connessione.
        """
        self.db_type = db_type
        self.collection_name = collection_name
        self.db = None

        if self.db_type == "milvus":
            self.db = Milvus(collection_name=self.collection_name, **kwargs)
        elif self.db_type == "quadrant":
            # Da implementare: Quadrant vectorstore
            pass
        else:
            raise ValueError("Database vettoriale non supportato.")

    def get_relevant_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Recupera i chunk più rilevanti per una query.
        Args:
            query (str): La domanda o il testo di ricerca.
            top_k (int): Numero di risultati da restituire.
        Returns:
            List[Document]: Lista di chunk/documenti rilevanti.
        """
        if not self.db:
            raise RuntimeError("Database vettoriale non inizializzato.")
        results = self.db.similarity_search(query, k=top_k)
        return results

# Esempio di utilizzo:
retriever = Retriever(db_type="milvus", collection_name="rag_chunks")
docs = retriever.get_relevant_documents("What is Task Decomposition?", top_k=3)
for doc in docs:
    print(doc.metadata["source"], doc.page_content)