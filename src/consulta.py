from config import *
from prompts import *
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever,EnsembleRetriever
from functools import lru_cache

def inicializar_rag():


    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings   # ojo: aquí el parámetro se llama embedding_function
    )


    #Retriever base con MMR
    mmr_retriever=vector_store.as_retriever( search_type=SEARCH_TYPE_MMR,
                                            search_kwargs={"k": SEARCH_K,
                                                            "lambda_mult":MMR_DIVERSITY_LAMBDA,
                                                            "fetch_k":MMR_FETCH_K})
    #Retriever adiccional con similarity
    similarity_retriever=vector_store.as_retriever(
        search_type=SEARCH_TYPE_SIMILARITY,
        search_kwargs={"k":SEARCH_K}
    )


    multi_query_prompt=PromptTemplate.from_template(MULTI_QUERY_PROMPT)


    multi_retriever=MultiQueryRetriever.from_llm(retriever=mmr_retriever,
                                        llm=llm_query,
                                        prompt=multi_query_prompt)
    
    #Ensemble Retriever
    if ENABLE_HYBRID_SEARCH:
        enseble_retriever=EnsembleRetriever(
            retrievers=[multi_retriever,similarity_retriever],
            weights=[0.7,0.3]
        )
        final_retriever=enseble_retriever
    else:final_retriever=multi_retriever


    prompt=PromptTemplate.from_template(RAG_TEMPLATE)



    # Funcion para formatear y preprocesar los documentos recuperados
    def format_docs(docs):
        formatted = []

        for i, doc in enumerate(docs, 1):
            header = f"[Fragmento {i}]"
            
            if doc.metadata:
                if 'source' in doc.metadata:
                    source = doc.metadata['source'].split("\\")[-1] if '\\' in doc.metadata['source'] else doc.metadata['source']
                    header += f" - Fuente: {source}"
                if 'page' in doc.metadata:
                    header += f" - Pagina: {doc.metadata['page']}"
        
            content = doc.page_content.strip()
            formatted.append(f"{header}\n{content}")
        
        return "\n\n".join(formatted)

    # La generacion parte de los documentos YA recuperados (clave "docs"),
    # no vuelve a llamar al retriever
    generar_respuesta = (
        {
            "context": itemgetter("docs") | RunnableLambda(format_docs),
            "question": itemgetter("question")
        }
        | prompt
        | llm_generation
        | StrOutputParser()
    )

    # El retriever se ejecuta UNA sola vez; esos mismos docs alimentan
    # la generacion y la salida de fuentes
    rag_chain = (
        RunnableParallel(
            docs=final_retriever,
            question=RunnablePassthrough()
        )
        | RunnablePassthrough.assign(respuesta=generar_respuesta)
    )

    return rag_chain

def query_rag(question):
    try:
        rag_chain = inicializar_rag()

        # Una sola recuperacion: respuesta y documentos salen del mismo invoke
        resultado = rag_chain.invoke(question)
        response = resultado["respuesta"]
        docs = resultado["docs"]

        # Formatear los documentos para mostrar
        docs_info = []
        for i, doc in enumerate(docs[:SEARCH_K], 1):
            doc_info = {
                "fragmento": i,
                "contenido": doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
                "fuente": doc.metadata.get('source', 'No especificada').split("\\")[-1],
                "pagina": doc.metadata.get('page', 'No especificada')
            }
            docs_info.append(doc_info)
        
        return response, docs_info
    
    except Exception as e:
        error_msg = f"Error al procesar la consulta: {str(e)}"
        return error_msg, []
    
def get_retriever_info():
    """Obtiene información sobre la configuración del retriever"""
    return {
        "tipo": f"{DEFAULT_SEARCH.upper()} + MultiQuery" + (" + Hybrid" if ENABLE_HYBRID_SEARCH else ""),
        "documentos": SEARCH_K,
        "diversidad": MMR_DIVERSITY_LAMBDA,
        "candidatos": MMR_FETCH_K,
        "umbral": SIMILARITY_THRESHOLD if ENABLE_HYBRID_SEARCH else "N/A"
    }
