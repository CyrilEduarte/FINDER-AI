import streamlit as st
import torch
import os
import chromadb
import uuid
import base64
import numpy as np
import urllib.parse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from llama_index.core import SimpleDirectoryReader
import PyPDF2

# Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FLAN-T5 Model for summarization
model_name = "google/flan-t5-large"  # Upgraded to Large for better summarization
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "cpu"
model.to(device)

# Ensure documents directory exists
doc_dir = "C:/Users/Administrator/Documents/python/documents"
os.makedirs(doc_dir, exist_ok=True)

# Function to extract and clean text from PDFs
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

# Function to extract key sections (first page and last 500 characters)
def extract_key_sections(text):
    lines = text.split("\n")
    first_page = " ".join(lines[:10])  # First 10 lines (Title/Header)
    last_part = text[-500:] if len(text) > 500 else text  # Last 500 chars
    return f"{first_page}\n{last_part}"

# Function to summarize text into a document description
def summarize_text(text, max_length=40):
    refined_prompt = f"""
    Identify the document type and its purpose in a concise sentence.
    Document Content:
    {text[:1000]}  # Limiting to the first 1000 characters for better context
    """.strip()
    
    inputs = tokenizer(refined_prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    summary_ids = model.generate(**inputs, max_length=max_length, min_length=10, length_penalty=2.0)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Streamlit UI
st.title("📄 SVHFI | FINDER-AI")
st.subheader("Fast Intelligent Navigation for Document Extraction & Retrieval")
st.sidebar.header("Upload PDFs")

# Upload Documents
uploaded_files = st.sidebar.file_uploader("Upload Documents", accept_multiple_files=True, type=["pdf"])

# Save uploaded files to disk
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(doc_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())  # Save the file permanently
    st.sidebar.success("Files saved! Click 'Process Documents' to index them.")

# Index Documents in ChromaDB
if st.sidebar.button("Process Documents"):
    st.sidebar.info("Processing documents...")
    documents = SimpleDirectoryReader(input_dir=doc_dir).load_data()
    collection = chroma_client.get_or_create_collection(name="business_docs")
    
    for doc in documents:
        key_sections = " ".join(doc.text.split("\n\n")[:5])  # Extract key sections
        summary = summarize_text(key_sections)  # Generate concise summary
        embeddings = [embedding_model.encode(summary).tolist()]
        doc_id = str(uuid.uuid4())

        collection.add(
            ids=[doc_id],
            documents=[summary],  
            embeddings=embeddings,  
            metadatas=[{"filename": doc.metadata.get("file_name", os.path.basename(doc.metadata.get("name", "Unknown")))}]
        )

    st.sidebar.success("Documents Indexed and Saved to ChromaDB!")

# Delete Files
delete_file = st.sidebar.selectbox("Select a file to delete:", os.listdir(doc_dir) if os.listdir(doc_dir) else ["No files available"])
if st.sidebar.button("Delete Selected File") and delete_file != "No files available":
    os.remove(os.path.join(doc_dir, delete_file))
    collection = chroma_client.get_or_create_collection(name="business_docs")
    collection.delete(where={"filename": delete_file})  # Remove from index
    st.sidebar.success(f"Deleted {delete_file}")
    st.rerun()

# 🔄 Re-Index All Files Button
if st.sidebar.button("🔄 Re-Index All Files"):
    st.sidebar.info("Re-indexing documents. This may take some time...")

    # Clear existing ChromaDB collection correctly
    collection = chroma_client.get_or_create_collection(name="business_docs")
    existing_docs = collection.get()

    if "ids" in existing_docs and existing_docs["ids"]:
        collection.delete(ids=existing_docs["ids"])  # Delete all existing document IDs

    # Re-index all documents from scratch
    for filename in os.listdir(doc_dir):
        file_path = os.path.join(doc_dir, filename)
        text = extract_text_from_pdf(file_path)
        if not text:
            continue  # Skip empty documents

        key_sections = extract_key_sections(text)  # Extract key sections
        summary = summarize_text(key_sections)  # Generate new summary
        embeddings = [embedding_model.encode(summary).tolist()]
        doc_id = str(uuid.uuid4())

        collection.add(
            ids=[doc_id],
            documents=[summary],  
            embeddings=embeddings,  
            metadatas=[{"filename": filename}]
        )

    st.sidebar.success("Re-indexing complete! All files have been processed again.")
    st.rerun()  # Refresh Streamlit to reflect changes

# Retrieve Stored Files in ChromaDB
collection = chroma_client.get_or_create_collection(name="business_docs")
existing_docs = collection.get()

# Extract unique filenames from metadata
stored_filenames = set()
if existing_docs and "metadatas" in existing_docs and existing_docs["metadatas"]:
    stored_filenames = {meta["filename"] for meta in existing_docs["metadatas"] if "filename" in meta}

# Display stored files in Streamlit
st.sidebar.subheader("📂 Files in Database")
if stored_filenames:
    for filename in sorted(stored_filenames):
        st.sidebar.write(f"- {filename}")
else:
    st.sidebar.write("No files stored yet.")

# Document Search Query
description = st.text_input("🔍 Describe the document you are looking for:")

if description:
    collection = chroma_client.get_or_create_collection(name="business_docs")
    query_embedding = embedding_model.encode(description).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=10)
    
    # Manually filter filenames that contain the search text
    matching_files = []
    for metadata in results["metadatas"][0]:  
        if "filename" in metadata and description.lower() in metadata["filename"].lower():
            matching_files.append(metadata["filename"])
    
    # Add additional results from filename matches
    if matching_files:
        for filename in matching_files:
            file_path = os.path.join(doc_dir, filename)
            with st.expander(f"📄 {filename} (Matched by Filename)"):
                st.markdown(f"[📄 Open {filename}](./documents/{filename})", unsafe_allow_html=True)


    st.subheader("📄 Potential Matching Documents")
    response_filenames = {}
    summaries = {}

    if "documents" in results and results["documents"]:
        for metadata, doc, distance in zip(results["metadatas"][0], results["documents"][0], results["distances"][0]):
            confidence = round(100 - distance, 2)
            filename = metadata.get('filename', 'Unknown')
            response_filenames[filename] = max(response_filenames.get(filename, 0), confidence)
            summaries[filename] = doc  # Store summary
        
        for filename, confidence in sorted(response_filenames.items(), key=lambda x: -x[1]):
            file_path = os.path.join(doc_dir, filename)
            
            # Ensure file URL encoding for spaces and special characters
            # file_url = f"file:///{urllib.parse.quote(file_path)}"
            with open(file_path, "rb") as f:
                file_data = f.read()


            with st.expander(f"📄 {filename} (Confidence: {confidence}%)"):
                st.markdown(f"**Summary:** {summaries[filename]}")
                # st.markdown(f'<a href="{file_url}" target="_blank">📄 Open {filename} in a new tab</a>', unsafe_allow_html=True)
                st.download_button(label=f"📥 Download {filename}", data=file_data, file_name=filename, mime="application/pdf")
    else:
        st.write("No relevant documents found.")
