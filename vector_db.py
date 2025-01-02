import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
import rag_manager
import glob
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

def run(chroma_path: str, data_path: str, model_type: str, scan_type: str = "full"):
    if scan_type == "reset":
        print("Clearing Database")
        clear_database(chroma_path)
    elif scan_type == "modify":
        print("Loading document")
        document = load_document(data_path)
        print("Splitting document into chunks")
        chunks = split_documents(document)
        print("Modifying documents in Chroma database...")
        modify_to_chroma(chunks, chroma_path, model_type)
        print("Database updated successfully.")

    elif scan_type == "remove":
        print(f"Removing document: {data_path}")
        remove_from_chroma(data_path, chroma_path)  # Pass the file path
        print("Database updated successfully.")
        
    else:
        print("Loading documents...")
        documents = load_folder(data_path)

        print("Splitting documents into chunks...")
        chunks = split_documents(documents)

        print("Adding chunks to Chroma database...")
        add_to_chroma(chunks, chroma_path, model_type)
        print("Database updated successfully.")

def load_document(data_path):
    """
    Load a single document from the data directory. Handles PDFs and text-based files.
    Extend this to handle more file types as needed.
    """

    # Load Text (txt, md, etc.)
    if data_path.endswith(".txt") or data_path.endswith(".md"):
        text_loader = TextLoader(data_path, encoding="utf-8")
        text_docs = text_loader.load()
        return text_docs
    
    # Load DOCX
    if data_path.endswith(".docx"):
        docx_loader = Docx2txtLoader(data_path)
        docx_docs = docx_loader.load()
        return docx_docs
    
    # Load PPTX
    if data_path.endswith(".pptx"):
        ppt_loader = UnstructuredPowerPointLoader(data_path)
        ppt_docs = ppt_loader.load()
        return ppt_docs
    # Load PDFs
    elif data_path.endswith(".pdf"):
        pdf_loader = PyPDFLoader(data_path)
        pdf_docs = pdf_loader.load()
        return pdf_docs
    else:
        raise ValueError(f"Unsupported file type: {data_path}")
def load_folder(data_path):
    """
    Load documents from the data directory recursively. Handles PDFs and text-based files.
    Extend this to handle more file types as needed.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path '{data_path}' does not exist.")

    all_docs = []

    # Traverse the directory tree
    for root, dirs, files in os.walk(data_path):
        # Process PDFs
        pdf_files = [os.path.join(root, f) for f in files if f.endswith('.pdf')]
        for pdf_file in pdf_files:
            try:
                pdf_loader = PyPDFLoader(pdf_file)
                pdf_docs = pdf_loader.load()
                all_docs.extend(pdf_docs)
            except Exception as e:
                print(f"Error loading PDF {pdf_file}: {e}")

        # Process Text files (txt, md, etc.)
        text_files = [os.path.join(root, f) for f in files if f.endswith(('.txt', '.md'))]
        for txt_file in text_files:
            try:
                text_loader = TextLoader(txt_file, encoding="utf-8")
                text_docs = text_loader.load()
                all_docs.extend(text_docs)
            except Exception as e:
                print(f"Error loading text file {txt_file}: {e}")

    return all_docs

def split_documents(documents: list[Document]):
    """Split documents into smaller chunks using a text splitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document], chroma_path: str, model_type: str):
    """Add document chunks to the Chroma vector database."""
    db = Chroma(persist_directory=chroma_path,
                embedding_function=rag_manager.select_embeddings(model_type))

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add.")

def modify_to_chroma(chunks: list[Document], chroma_path: str, model_type: str):
    """Modify existing document chunks in the Chroma vector database."""
    db = Chroma(persist_directory=chroma_path,
                embedding_function=rag_manager.select_embeddings(model_type))

    new_chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=["metadatas"])  # Corrected to "metadatas"
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Remove existing chunks with the same source
    sources_to_remove = {chunk.metadata.get("source") for chunk in new_chunks_with_ids if "source" in chunk.metadata}
    chunks_to_remove = [
        item["id"] for item in existing_items["metadatas"]
        if item.get("source") in sources_to_remove
    ]
    
    if chunks_to_remove:
        print(f"Removing existing chunks: {len(chunks_to_remove)}")
        db.delete(ids=chunks_to_remove)

    # Add new chunks
    new_chunks = [chunk for chunk in new_chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add.")



def remove_from_chroma(document_path: str, chroma_path: str):
    """Remove document chunks from the Chroma vector database."""
    db = Chroma(persist_directory=chroma_path)
    existing_items = db.get(include=["metadatas"])  # Corrected to "metadatas"
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Check and remove items matching the document source
    chunks_to_remove = [
        item["id"] for item in existing_items["metadatas"]
        if item.get("source") == document_path
    ]
    
    if chunks_to_remove:
        print(f"Removing existing chunks: {len(chunks_to_remove)}")
        db.delete(ids=chunks_to_remove)
    else:
        print(f"No chunks found for source: {document_path}")



def calculate_chunk_ids(chunks: list[Document]):
    """Assign unique IDs to each chunk based on the source, page, and chunk index."""
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks

def clear_database(chroma_path):
    """Remove the existing Chroma database directory."""
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_type", type=str, required=True, choices=["reset", "modify", "remove", "full"], help="Specify the scan type: reset, modify, remove, full.")
    parser.add_argument("--chroma_path", type=str, required=True, help="Path to the Chroma database directory")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory or file")
    parser.add_argument("--model_type", type=str, required=True, help="Type of embeddings to use")
    args = parser.parse_args()

    run(args.chroma_path, args.data_path, args.model_type, args.scan_type)
