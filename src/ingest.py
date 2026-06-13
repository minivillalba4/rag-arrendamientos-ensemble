from config import *
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


document_loader=PyPDFDirectoryLoader(UBICACION_PDFS)
documentos=document_loader.load()
print(f"se cargadon {len(documentos)} paginas de documentos")

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1200 ,chunk_overlap=450)

splits=text_splitter.split_documents(documents=documentos)

print(len(splits))

print(f"se crearon {len(splits)} chunks de texto")


vector_store=Chroma.from_documents(documents=splits,
                                   embedding=embeddings,
                                   persist_directory=PERSIST_DIR)

print(f"Base de datos vectorial creada en {PERSIST_DIR}")