import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI,OpenAIEmbeddings

load_dotenv()

# Verificamos que la clave esté configurada (langchain_openai la lee de OPENAI_API_KEY)
GPT_API_KEY = os.environ["OPENAI_API_KEY"]

# 2. Inicializamos el LLM (el modelo que generará las respuestas)
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# 3. Inicializamos el Embedding (el modelo que convertirá tus documentos en vectores)
EMBEDDING = OpenAIEmbeddings(model="text-embedding-3-small")