import os # non so se ralmente necessaia
from langchain_community.document_loaders import 
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentLoader:
    """
    Loader modulare per documenti PDF.
    Usa Langchain per caricare e suddividere i documenti in chunk.
    """

    def __init__(self, docs_dir: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Args:
            docs_dir (str): Directory contenente i PDF.
            chunk_size (int): Dimensione massima di ogni chunk.
            chunk_overlap (int): Sovrapposizione tra chunk.
        """
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self):
        """
        Carica tutti i PDF dalla directory.
        Returns:
            List[Document]: Lista di documenti Langchain.
        """
        documents = []
        for filename in os.listdir(self.docs_dir):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(self.docs_dir, filename)
                loader = PDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
        return documents

    def chunk_documents(self, documents):
        """
        Suddivide i documenti in chunk.
        Args:
            documents (List[Document]): Documenti da suddividere.
        Returns:
            List[Document]: Lista di chunk.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        return chunks

    def load_and_chunk(self):
        """
        Carica e suddivide i documenti in un'unica funzione.
        Returns:
            List[Document]: Lista di chunk.
        """
        docs = self.load_documents()
        chunks = self.chunk_documents(docs)
        return chunks

# Esempio di utilizzo:
# loader = DocumentLoader(docs_dir="docs/")
# chunks = loader.load_and_chunk()