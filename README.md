# Conversational Research Document Analyzer

A multi-turn conversational AI assistant that lets users upload PDF documents — including research papers — and query their content using natural language, with full conversational memory across follow-up questions.

## Overview

Reading long PDFs or research papers to find specific information is slow. This project solves that by combining a Retrieval-Augmented Generation (RAG) pipeline with conversational memory, so users can "chat" with a document: ask a question, get a grounded answer pulled from the actual text, then ask a follow-up that still has context from earlier in the conversation.

## Features

- **PDF Ingestion** — Upload and parse research papers or any PDF document for analysis.
- **RAG Pipeline** — Combines semantic retrieval with LLM generation so answers are grounded in the document's actual content, not just model memory.
- **Semantic Search** — Document embeddings stored in a Chroma vector database, enabling accurate similarity-based retrieval over large documents.
- **Conversational Memory** — Maintains chat history across turns, so follow-up questions are interpreted in context rather than in isolation.
- **Context-Aware Responses** — Generates answers using only the most relevant retrieved chunks, improving accuracy and reducing hallucination.

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain |
| LLM | OpenAI |
| Vector Database | Chroma |
| Embeddings | OpenAI Embeddings |
| Language | Python |

## How It Works

1. **Ingest** — The PDF is parsed and split into chunks suitable for embedding.
2. **Embed & Store** — Each chunk is embedded and stored in Chroma for fast semantic retrieval.
3. **Query** — A user's question is embedded and matched against the stored chunks to find the most relevant context.
4. **Generate** — The retrieved context, plus prior conversation history, is passed to the LLM to generate a grounded, context-aware response.
5. **Remember** — Conversational memory updates after each turn, allowing natural follow-up questions.

## Getting Started

### Prerequisites
- Python 3.10+
- OpenAI API key

### Installation
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### Configuration
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_api_key_here
```

### Running the App
```bash
python main.py
```

Then upload a PDF and start asking questions in natural language.

## Example Usage

```
> Upload: attention_is_all_you_need.pdf

User: What problem does this paper address?
Assistant: The paper addresses the limitations of recurrent and convolutional
models for sequence transduction by proposing the Transformer architecture...

User: How does it compare to RNNs in training speed?
Assistant: Building on the previous point, the paper shows the Transformer
trains significantly faster than RNN-based models because...
```

## Project Structure
```
.
├── ingestion/        # PDF parsing and chunking
├── embeddings/        # Embedding generation and Chroma storage
├── retrieval/          # Semantic search logic
├── memory/             # Conversational memory management
├── main.py             # Entry point
└── requirements.txt
```

## What I Learned

This project deepened my understanding of how to design and integrate real LLM applications end-to-end — from chunking and embedding strategy, to vector search tuning, to managing conversational state across turns. It reflects practical experience with RAG architecture, vector databases, and prompt/context engineering, which are core skills for applied AI engineering roles.

## Author

**Ubrillo** — [GitHub](https://github.com/<your-username>) · [LinkedIn](#)
