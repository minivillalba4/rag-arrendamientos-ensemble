from langchain_core.prompts import ChatPromptTemplate

prompt_rag = ChatPromptTemplate.from_messages(
    [
        ("system", "Eres un asistente que responde preguntas sobre un contrato de "
                   "arrendamiento. Responde únicamente con la información del contexto. "
                   "Si la respuesta no está en el contexto, di que no lo sabes; no inventes."),
        ("human", "{pregunta}. Adjunto contexto: {contexto}"),
    ]
)
