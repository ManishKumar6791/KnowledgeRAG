import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import torch
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_community.llms import Ollama

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker


class DocumentProcessor:
    def __init__(self):
        self.setup_document_converter()
        self.setup_local_components()

        # Path for persistent vector database
        self.db_path = Path("vector_db/docling.db")
        self.index_name = "document_chunks"

        # Initialize ChromaDB
        print("Initializing ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.chroma_client.get_or_create_collection(name=self.index_name)

    def setup_document_converter(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.ocr_options.lang = ["en"]
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.AUTO
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )

    def setup_local_components(self):
        self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.llm = Ollama(model="llama3", temperature=0.1)
        self.tokenizer = "jinaai/jina-embeddings-v3"
        self.max_tokens = 512

    def extract_chunk_metadata(self, chunk) -> Dict[str, Any]:
        metadata = {
            "text": chunk.text,
            "headings": [],
            "page_info": None,
            "content_type": None
        }
        if hasattr(chunk, 'meta'):
            if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                metadata["headings"] = chunk.meta.headings
            if hasattr(chunk.meta, 'doc_items'):
                for item in chunk.meta.doc_items:
                    if hasattr(item, 'label'):
                        metadata["content_type"] = str(item.label)
                    if hasattr(item, 'prov') and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, 'page_no'):
                                metadata["page_info"] = prov.page_no
        return metadata

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        print(f"Processing PDF: {pdf_path}")
        result = self.converter.convert(pdf_path)
        doc = result.document
        chunker = HybridChunker(tokenizer=self.tokenizer, max_tokens=self.max_tokens)
        chunks = list(chunker.chunk(doc))

        processed_chunks = []
        for chunk in chunks:
            metadata = self.extract_chunk_metadata(chunk)
            metadata['source_file'] = os.path.basename(pdf_path)
            processed_chunks.append(metadata)
        return processed_chunks

    def process_web_content(self, url: str) -> List[Dict[str, Any]]:
        print(f"Scraping URL: {url}")
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch {url}: {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        # Extract all visible text
        texts = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
        chunks = []
        for i, text in enumerate(texts):
            chunks.append({
                "text": text,
                "headings": [],
                "page_info": None,
                "content_type": "web",
                "source_file": url
            })
        return chunks

    def process_documents(self, input_path: str):
        all_chunks = []

        # Handle PDF files or directories
        if os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
            all_chunks.extend(self.process_pdf(input_path))
        elif os.path.isdir(input_path):
            for filename in os.listdir(input_path):
                if filename.lower().endswith(".pdf"):
                    all_chunks.extend(self.process_pdf(os.path.join(input_path, filename)))
        elif input_path.startswith("http"):  # Treat as URL
            all_chunks.extend(self.process_web_content(input_path))
        else:
            raise ValueError("Input must be a PDF file, folder of PDFs, or a URL starting with http/https.")

        # Prepare embeddings
        documents = []
        embeddings = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(all_chunks):
            text = chunk['text']
            documents.append(text)
            embeddings.append(self.embed_model.encode(text))
            metadatas.append({
                "headings": json.dumps(chunk.get("headings", [])),
                "page": str(chunk.get("page_info")),
                "content_type": str(chunk.get("content_type")),
                "source_file": str(chunk.get("source_file"))
            })
            ids.append(f"{chunk.get('source_file', 'doc')}_{i}")

        # Store in ChromaDB
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Stored {len(documents)} chunks in ChromaDB.")

    def query(self, question: str, k: int = 5) -> str:
        query_embedding = self.embed_model.encode(question)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas"]
        )

        chunks = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunk = {
                "text": doc,
                "headings": json.loads(meta.get("headings", "[]")),
                "page": meta.get("page"),
                "content_type": meta.get("content_type"),
                "source_file": meta.get("source_file")
            }
            chunks.append(chunk)

        context = self.format_context(chunks)
        prompt = f"""You are a helpful assistant. Use the following document excerpts to answer the user's question.

- Keep the answer brief, clear, and easy to understand.
- Format your answer using bullet points or numbered lists when appropriate.
- Do **not** mention page numbers or section headings.
- Do **not** repeat the excerpts directly. Summarize in your own words.

Document Excerpts:
{context}

Question: {question}

Answer:"""
        return self.llm(prompt)

    def format_context(self, chunks: List[Dict]) -> str:
        parts = []
        for chunk in chunks:
            try:
                if chunk['headings']:
                    parts.append(f"Section: {' > '.join(chunk['headings'])}")
            except Exception:
                pass
            if chunk['page']:
                parts.append(f"Page {chunk['page']}:")
            parts.append(chunk['text'])
            parts.append("-" * 40)
        return "\n".join(parts)

