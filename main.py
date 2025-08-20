import gradio as gr
from processor import KnowledgeRAGProcessor

# Paths
PDF_FOLDER = "docs"
URL_FILE = "urls.txt"

# Initialize processor
processor = KnowledgeRAGProcessor()

# Load existing collection or process PDFs/URLs
processor.load_or_update(pdf_folder=PDF_FOLDER, url_file=URL_FILE)

# Gradio query function
def query_knowledge(question):
    if not question.strip():
        return "Please enter a question."
    return processor.query(question)

# Launch Gradio
iface = gr.Interface(
    fn=query_knowledge,
    inputs=gr.Textbox(lines=2, placeholder="Ask anything about your documents or web content..."),
    outputs=gr.Textbox(label="Answer"),
    title="KnowledgeRAG Assistant",
    description="Query your PDFs and web content using LLM + RAG."
)
iface.launch()
