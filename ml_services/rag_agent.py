import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.schema import Document
import sys
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Use API Key from settings
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

from langchain.document_loaders import DirectoryLoader, TextLoader
import glob

class InternalSalesAgent:
    def __init__(self):
        self.vector_store = None
        self.documents = []
        # Lazy load or auto-load on init
        self.ingest_docs()

    def ingest_docs(self):
        docs_dir = os.path.join(settings.DATA_DIR, "docs")
        if not os.path.exists(docs_dir):
            logger.warning(f"Docs directory not found: {docs_dir}")
            return

        logger.info(f"Loading documents from {docs_dir}...")
        try:
            # Simple text loader for .md files
            loader = DirectoryLoader(docs_dir, glob="**/*.md", loader_cls=TextLoader)
            self.documents = loader.load()
            
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            self.docs_chunks = splitter.split_documents(self.documents)
            
            logger.info(f"Ingested {len(self.documents)} files, split into {len(self.docs_chunks)} chunks.")
            
            # Here we would init the VectorStore with Embeddings
            # self.vector_store = FAISS.from_documents(self.docs_chunks, OpenAIEmbeddings())
            
        except Exception as e:
            logger.error(f"Error ingesting docs: {e}")

    def query(self, question):
        """
        RAG Query - Simple retrieval simulation for POC
        """
        if not self.documents:
            return "Knowledge base is empty. Please check data/docs directory."
            
        # Mock retrieval: Check if keywords exist in docs
        # In real RAG: relevant_docs = self.vector_store.similarity_search(question)
        
        logger.info(f"Searching knowledge base for: {question}")
        
        # Simple keyword search for POC demonstration without embeddings cost
        relevant_chunks = []
        keywords = question.lower().split()
        for doc in self.docs_chunks:
            content = doc.page_content.lower()
            score = sum(1 for k in keywords if k in content)
            if score > 0:
                relevant_chunks.append((score, doc.page_content))
                
        if not relevant_chunks:
             return "I couldn't find specific information in the policy documents."
             
        # Return top match
        relevant_chunks.sort(key=lambda x: x[0], reverse=True)
        best_match = relevant_chunks[0][1]
        
        return f"Based on internal policies:\n\n...{best_match[:300]}...\n\n(Source: Internal Knowledge Base)"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = InternalSalesAgent()
    print(agent.query("What is the return policy?"))
