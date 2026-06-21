# RAG Arrendamientos (Ensemble)

Sistema RAG (Retrieval-Augmented Generation) para responder preguntas sobre
contratos de arrendamiento a partir de documentos PDF. Usa LangChain, Chroma
como base de datos vectorial y un *ensemble retriever* (MultiQuery + MMR).

## Requisitos

- Python 3.10+
- Una clave de API de OpenAI

## Instalación

```bash
# 1. Crear y activar un entorno virtual
python -m venv .venv
.venv\Scripts\activate        # En Windows
# source .venv/bin/activate   # En macOS/Linux

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar las claves
copy .env.example .env        # En Windows (cp en macOS/Linux)
# Edita .env y pon tu clave real de OPENAI_API_KEY
```

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
