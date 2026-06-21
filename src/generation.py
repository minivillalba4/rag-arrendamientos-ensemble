from langchain_chroma import Chroma
from config import *
from langchain_classic.retrievers import MultiQueryRetriever,EnsembleRetriever
from prompt import prompt_rag
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


vector_store=Chroma(embedding_function=EMBEDDING ,
                    persist_directory="./data/bbdd_vect")

similarity_retriever=vector_store.as_retriever(search_type="similarity",
                                         search_kwargs={"k":3})
multiquery_retriever=MultiQueryRetriever.from_llm(retriever=similarity_retriever,
                    llm=LLM)

mmr_retriever=vector_store.as_retriever(search_type="mmr",
                                        search_kwargs={"k":3,
                                                      "fetch_k": 10, "lambda_mult":0.67})

ensemble_retriever=EnsembleRetriever(retrievers=[multiquery_retriever,mmr_retriever],
                                     weights=[0.6,0.4],c=3
                                     )


def formatear_contexto(docs):
    return "\n\n".join(
        f"[{i}] {doc.metadata.get('source', 'No especificado')} "
        f"(pág. {doc.metadata.get('page', 'No especificado')})\n{doc.page_content}"
        for i, doc in enumerate(docs[:3], 1)
    )


rag_chain=(
    {
        "pregunta":RunnablePassthrough(),
        "contexto":ensemble_retriever | formatear_contexto
    }
    |prompt_rag|LLM|StrOutputParser()
)


pregunta="De que tratan los documentos?"
contexto=ensemble_retriever.invoke(pregunta)
respuesta=rag_chain.invoke(pregunta)
print(respuesta,"\n")
print("-----------------------------------")
print(formatear_contexto(contexto))
