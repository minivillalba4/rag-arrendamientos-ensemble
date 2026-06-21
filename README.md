# RAG Arrendamientos (Ensemble)

Sistema RAG (Retrieval-Augmented Generation) para responder preguntas sobre
contratos de arrendamiento a partir de documentos PDF. Usa LangChain, Chroma
como base de datos vectorial y un *ensemble retriever* (MultiQuery + MMR).

## Requisitos

- Python 3.10+
- Una clave de API de OpenAI


## Uso

1. Coloca tus PDFs en la carpeta `data/`.
2. Genera la base de datos vectorial (indexa los documentos):

   ```bash
   python src/ingest.py
   ```

3. Ejecuta las consultas RAG:

   ```bash
   python src/generation.py
   ```

## Estructura

- `src/config.py` — configuración del LLM y los embeddings.
- `src/ingest.py` — carga los PDFs y crea la base de datos vectorial Chroma.
- `src/prompt.py` — plantilla del prompt para el modelo.
- `src/generation.py` — cadena RAG con el ensemble retriever.

## Notas

- El archivo `.env` y la carpeta `data/` están en `.gitignore` y **no** se suben al repositorio.
