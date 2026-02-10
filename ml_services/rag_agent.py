import os
import sys
import logging
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import RetrievalQA

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Use API Key from settings
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

class InternalSalesAgent:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.index_path = os.path.join(settings.DATA_DIR, "faiss_index")
        
        # Initialize Embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Load or Create Index
        self.ingest_docs()

    def ingest_docs(self):
        docs_dir = os.path.join(settings.DATA_DIR, "docs")
        if not os.path.exists(docs_dir):
            logger.warning(f"Docs directory not found: {docs_dir}")
            return

        # Check if index exists (simple caching)
        if os.path.exists(self.index_path):
            try:
                logger.info("Loading existing FAISS index...")
                self.vector_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
                self.setup_chain()
                return
            except Exception as e:
                logger.warning(f"Failed to load index: {e}. Rebuilding...")

        logger.info(f"Loading documents from {docs_dir}...")
        try:
            # Load Docs
            loader = DirectoryLoader(docs_dir, glob="**/*.md", loader_cls=TextLoader)
            documents = loader.load()
            
            if not documents:
                logger.warning("No documents found.")
                return

            # Split Docs
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs_chunks = splitter.split_documents(documents)
            
            logger.info(f"Ingested {len(documents)} files, split into {len(docs_chunks)} chunks.")
            
            # Create Index
            self.vector_store = FAISS.from_documents(docs_chunks, self.embeddings)
            self.vector_store.save_local(self.index_path)
            
            self.setup_chain()
            
        except Exception as e:
            logger.error(f"Error ingesting docs: {e}", exc_info=True)

    def setup_chain(self):
        if not self.vector_store:
            return
            
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )

    def query(self, question):
        """
        RAG Query using RetrievalQA
        """
        if not self.qa_chain:
            return "Knowledge base is likely empty or failed to load. Please check data/docs."
            
        logger.info(f"RAG Query: {question}")
        try:
            response = self.qa_chain.run(question)
            return response
        except Exception as e:
            logger.error(f"RAG Error: {e}")
            return f"I encountered an error retrieving that information: {e}"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = InternalSalesAgent()
    print(agent.query("What is the return policy?"))
