from loader import DocumentLoader
from embedder import Embedder

loader = DocumentLoader(docs_dir="docs/")
chunks = loader.load_and_chunk()

embedder = Embedder(embedder_type="azure_openai", db_type="milvus")
embedder.process(chunks)