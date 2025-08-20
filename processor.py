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

        try:
            if hasattr(chunk, 'meta'):
                # 1. Direct headings
                if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                    metadata["headings"] = chunk.meta.headings

                # 2. Structured doc_items
                if hasattr(chunk.meta, 'doc_items'):
                    for item in chunk.meta.doc_items:
                        if hasattr(item, 'label'):
                            metadata["content_type"] = str(item.label)
                        if hasattr(item, 'text') and item.label in ["Heading", "Title", "Subtitle"]:
                            metadata["headings"].append(item.text.strip())

                # 3. Provenance info (pages)
                if hasattr(chunk.meta, 'doc_items'):
                    for item in chunk.meta.doc_items:
                        if hasattr(item, 'prov') and item.prov:
                            for prov in item.prov:
                                if hasattr(prov, 'page_no'):
                                    metadata["page_info"] = prov.page_no

            # 4. Fallback: Regex guess
            if not metadata["headings"]:
                heading_guess = self.detect_heading(chunk.text)
                if heading_guess:
                    metadata["headings"] = [heading_guess]

        except Exception as e:
            print(f"Error extracting metadata: {e}")

        return metadata


    def detect_heading(self, text: str) -> str:
        import re
        lines = text.split("\n")
        for line in lines[:2]:
            if re.match(r"^\d+(\.\d+)*\s+[A-Z][A-Za-z\s]{2,}$", line.strip()):
                return line.strip()
            if line.isupper() and len(line.split()) < 10:
                return line.strip()
        return None
    
    def load_or_update(self, pdf_folder: str, url_file: str):
        """
        Load existing ChromaDB collection, or process PDFs/URLs if missing.
        
        - Checks if the ChromaDB collection exists.
        - If exists: loads collection.
        - If not: processes PDFs in `pdf_folder` and URLs in `url_file`.
        - Converts embeddings to lists and adds them to the collection.
        - Handles exceptions for PDFs, URLs, and ChromaDB operations.
        """
        try:
            # Check if ChromaDB exists
            chroma_exists = os.path.exists(self.db_path) and any(
                fname.endswith(".sqlite3") or fname.endswith(".db")
                for fname in os.listdir(self.db_path)
            )

            if chroma_exists:
                print("ChromaDB exists. Loading collection...")
                self.chroma_client = chromadb.PersistentClient(path=str(self.db_path))
                self.collection = self.chroma_client.get_or_create_collection(name=self.index_name)
                print(f"Loaded collection '{self.index_name}' with {self.collection.count()} records.")
            else:
                print("ChromaDB does not exist. Processing new documents and URLs...")
                
                # Process PDFs
                if os.path.exists(pdf_folder):
                    self.process_documents(pdf_folder)
                else:
                    print(f"PDF folder '{pdf_folder}' not found. Skipping PDF processing.")

                # Process URLs
                if os.path.exists(url_file):
                    with open(url_file, "r") as f:
                        urls = [line.strip() for line in f if line.strip()]
                    for url in urls:
                        try:
                            print(f"Processing URL: {url}")
                            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
                            if response.status_code == 200:
                                text_chunks = [p.get_text() for p in BeautifulSoup(response.text, "html.parser").find_all("p")]
                                for i, chunk_text in enumerate(text_chunks):
                                    if chunk_text.strip():
                                        embedding = self.embed_model.encode(chunk_text).tolist()
                                        self.collection.add(
                                            documents=[chunk_text],
                                            embeddings=[embedding],
                                            metadatas=[{"source_url": url}],
                                            ids=[f"{url}_{i}"]
                                        )
                            else:
                                print(f"Skipping URL {url}, status code: {response.status_code}")
                        except Exception as e:
                            print(f"Error processing URL {url}: {e}")
                else:
                    print(f"URL file '{url_file}' not found. Skipping URL processing.")

        except Exception as e:
            print(f"Error loading or updating ChromaDB: {e}")


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
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
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
        else:
            print(f"Failed to fetch page, status code: {response.status_code}")

       

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

