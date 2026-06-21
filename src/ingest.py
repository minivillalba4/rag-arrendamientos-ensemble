from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import *

loader=PyPDFDirectoryLoader("./data")
documents=loader.load()

print(len(documents))



splitter=RecursiveCharacterTextSplitter(chunk_size=700,
                                        chunk_overlap=250)
splits=splitter.split_documents(documents=documents)
print(len(splits))

vector_store=Chroma.from_documents(splits,
                                   embedding=EMBEDDING ,
                                   persist_directory="./data/bbdd_vect")

