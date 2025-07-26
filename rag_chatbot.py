import os
import logging
from logging.handlers import RotatingFileHandler
from time import sleep
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from google.generativeai import configure, GenerativeModel
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import shutil
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from flask_cors import CORS  # Added import for CORS

# Set telemetry and USER_AGENT environment variables early
os.environ["CHROMA_TELEMETRY_ENABLED"] = "FALSE"
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# Set up logging with telemetry filter
class TelemetryFilter(logging.Filter):
    def filter(self, record):
        return "Failed to send telemetry event" not in record.getMessage()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('chatbot.log', maxBytes=1000000, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.addFilter(TelemetryFilter())

# Log ChromaDB version
logger.info(f"Using ChromaDB version: {chromadb.__version__}")

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Added CORS with wildcard (*) to allow all origins

# Set up Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("Gemini API Key not provided in environment variables")
    raise ValueError("Gemini API Key not provided")
configure(api_key=GEMINI_API_KEY)
logger.info("Gemini API configured successfully")

# Step 1: Index website content using Selenium
def index_content():
    try:
        # Define base URL and SPA routes
        base_url = "https://campuslink-sece.vercel.app"
        routes = ["/","/announcements"]
        urls = [urljoin(base_url, route) for route in routes]
        logger.info(f"Loading content from {len(urls)} URLs: {urls}")

        # Set up headless Chrome
        options = Options()
        options.add_argument("--headless")
        options.add_argument(f"user-agent={os.environ['USER_AGENT']}")
        driver = webdriver.Chrome(options=options)
        documents = []
        for url in urls:
            try:
                driver.get(url)
                sleep(5)  # Increased wait time for JavaScript rendering
                content = driver.page_source
                soup = BeautifulSoup(content, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                print(f"[DEBUG] Scraped Page Text:\n\n{text}\n{'='*80}")
                if text.strip():
                    documents.append(Document(page_content=text, metadata={"source": url}))
                    logger.info(f"Loaded content from {url} ({len(text)} characters)")
                else:
                    logger.warning(f"No content extracted from {url}")
            except Exception as e:
                logger.warning(f"Failed to load {url}: {str(e)}")
        driver.quit()

        if not documents:
            logger.error("No content loaded from any URLs")
            raise ValueError("No content loaded from any URLs")

        logger.info(f"Loaded {len(documents)} documents from website")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)
        if not chunks:
            logger.error("No chunks generated after splitting documents")
            raise ValueError("No chunks generated after splitting documents")
        logger.info(f"Split documents into {len(chunks)} chunks")

        # Clear ChromaDB persistence directory
        persist_dir = "./data/chroma_db"
        if os.path.exists(persist_dir):
            logger.info(f"Clearing existing ChromaDB directory: {persist_dir}")
            shutil.rmtree(persist_dir)

        # Create embeddings and store in ChromaDB
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        client = chromadb.Client(Settings(anonymized_telemetry=False))
        logger.info("Initialized ChromaDB client with telemetry disabled")
        vector_store = Chroma.from_documents(
            chunks,
            embeddings,
            collection_name="website_data",
            persist_directory=persist_dir,
            client=client
        )
        logger.info(f"Indexed {len(chunks)} chunks into ChromaDB at {persist_dir}")
        return vector_store
    except Exception as e:
        logger.error(f"Error indexing website content: {str(e)}", exc_info=True)
        raise

# Step 2: Create RAG prompt
def make_rag_prompt(query, relevant_passage):
    try:
        escaped_passage = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
        prompt = f"""You are a friendly and helpful bot that answers questions using the provided content. 
        Answer in a concise, conversational tone suitable for all audiences. 
        Look for information related to the query, including synonyms or related events (e.g., Independence Day for flag hoisting).
        Use only the information from the passage. If the passage doesn't have the answer, say so politely.
        QUESTION: '{query}'
        PASSAGE: '{escaped_passage}'
        ANSWER: """
        logger.debug(f"Created prompt for query: {query}")
        return prompt
    except Exception as e:
        logger.error(f"Error creating RAG prompt: {str(e)}", exc_info=True)
        raise

# Step 3: Generate response using Gemini with retry logic
def generate_response(prompt, retries=3, delay=2):
    try:
        for attempt in range(retries):
            try:
                logger.info(f"Generating response with Gemini model gemini-1.5-flash (attempt {attempt + 1})")
                model = GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                logger.debug(f"[DEBUG] Gemini Response:\n\n{response.text}\n{'='*80}")
                return response.text
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < retries - 1:
                    sleep(delay)
                else:
                    logger.error(f"All {retries} attempts failed for Gemini response")
                    return "Sorry, I couldn't process your request right now."
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        return "Sorry, I couldn't process your request right now."

# Step 4: Process query
def generate_answer(vector_store, query):
    try:
        logger.info(f"Processing query: {query}")
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        relevant_docs = retriever.invoke(query)

        if not relevant_docs:
            logger.warning(f"No relevant documents found for query: {query}")
            return "Sorry, I couldn't find any relevant information for your query."

        relevant_text = "\n---\n".join([doc.page_content for doc in relevant_docs])
        logger.debug(f"[DEBUG] Retrieved Relevant Chunks for query '{query}':\n{relevant_text}\n{'='*80}")

        prompt = make_rag_prompt(query, relevant_text)
        answer = generate_response(prompt)
        logger.info(f"Generated answer: {answer[:100]}...")
        return answer
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return "Sorry, something went wrong while processing your query."

# Step 5: Flask endpoint for the chatbot
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        logger.debug(f"Received request data: {data}")
        query = data.get('query', '')
        
        if not query or len(query.strip()) == 0:
            logger.warning("Empty or invalid query received")
            return jsonify({"error": "Query cannot be empty"}), 400
        
        if len(query) > 500:
            logger.warning(f"Query too long: {len(query)} characters")
            return jsonify({"error": "Query is too long (max 500 characters)"}), 400

        answer = generate_answer(vector_store, query)
        logger.info(f"Returning response for query '{query}': {answer[:100]}...")
        return jsonify({"response": answer})
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# Initialize the vector store
try:
    logger.info("Initializing vector store")
    vector_store = index_content()
    logger.info("Vector store initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize vector store: {str(e)}", exc_info=True)
    raise

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True, host='0.0.0.0', port=5000)
