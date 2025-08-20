# KnowledgeRAG-Highlights-knowledge-retrieval-and-generation
KnowledgeRAG is a demo system that combines document processing, web scraping, and LLM-based summarization into a single Retrieval-Augmented Generation (RAG) workflow. Users can upload PDFs or provide web URLs, and the system indexes all content in ChromaDB. Queries retrieve the most relevant information and generate clear, concise answers.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-green)
![LLM](https://img.shields.io/badge/LLM-Ollama-orange)


---

## **Features**
- **PDF Support:** Extracts text from PDFs using OCR and table recognition.
- **Web Scraping:** Indexes content from any public web page.
- **Vector Search:** Uses ChromaDB to store embeddings and retrieve top-k relevant chunks.
- **LLM Summarization:** Generates concise, context-aware answers using Ollama.
- **Metadata Tracking:** Keeps headings, pages, content type, and source info.
- **Unified Pipeline:** Handles both PDFs and web pages seamlessly.

---

## **Tech Stack**
- Python 3.10+
- [PyTorch](https://pytorch.org/)
- [SentenceTransformers](https://www.sbert.net/)
- [ChromaDB](https://www.trychroma.com/)
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)
- [Ollama LLM](https://ollama.com/)
- [Docling](https://github.com/docling/docling)

---

## **Installation**
```bash
git clone https://github.com/yourusername/KnowledgeRAG.git
cd KnowledgeRAG
pip install -r requirements.txt
```

## **Quick Start (Run in One Line)**
```bash
python main.py
```

**Follow prompts to:**
- Provide a PDF file, folder of PDFs, or a web URL.
- Ask questions and get summarized answers.

**Example Usage**
- Enter file/folder path or URL: docs/sample1.pdf
- Enter your query: Summarize the main ideas about machine learning from these sources.

**Output:**
Concise, context-aware answer retrieved from PDF and/or web sources.

## **Demo Data**
- docs/ – Sample PDFs for testing
- examples/ – Sample web URLs
- vector_db/ – ChromaDB persistent storage (auto-populated)

```plaintext
Folder Structure
KnowledgeRAG/
│
├─ README.md
├─ requirements.txt
├─ main.py
├─ processor.py
├─ vector_db/
├─ docs/
├─ examples/
└─ .gitignore
```


