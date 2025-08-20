from processor import DocumentProcessor

def main():
    processor = DocumentProcessor()

    print("Welcome to KnowledgeRAG!")
    print("You can provide a PDF file, folder of PDFs, or a web URL (starting with http/https).")
    
    # Step 1: Process documents or URL
    input_path = input("Enter file/folder path or URL: ").strip()
    processor.load_or_process(input_path)

    # Step 2: Query loop
    while True:
        question = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        answer = processor.query(question)
        print("\nAnswer:")
        print(answer)
        print("-" * 80)

if __name__ == "__main__":
    main()
