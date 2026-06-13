from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()
#API KEYS
GEMINI_API_KEY=os.environ["GEMINI_API_KEY"]
OPEN_AI_API_KEY=os.environ["OPEN_AI_API_KEY"]

#RUTAS Y DIRECTORIOS
PERSIST_DIR=r".\data\persistencia_chroma"
UBICACION_PDFS="./data"

#MODELOS DE IA
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=GEMINI_API_KEY
)

llm_query = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=GEMINI_API_KEY
)
llm_generation=ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPEN_AI_API_KEY
)

#RETRIEVER MMR
SEARCH_TYPE_MMR="mmr"
MMR_DIVERSITY_LAMBDA=0.7
MMR_FETCH_K=20
SEARCH_K=3

#RETRIEVER SIMILARITY
SEARCH_TYPE_SIMILARITY="similarity"

#CONFIGURACIÓN ENSEMBLE RETRIEVER
ENABLE_HYBRID_SEARCH=True
SIMILARITY_THRESHOLD=0.7
DEFAULT_SEARCH=SEARCH_TYPE_MMR