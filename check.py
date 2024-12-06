import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline
import torch

# Load environment variables
load_dotenv()

pdf_path = "/home/khushi/faiss/eco1.pdf"

# Ensure the PDF file exists
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"The file '{pdf_path}' does not exist.")

# Load PDF and split into chunks
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Check if documents were successfully loaded
if not documents:
    raise ValueError("No documents were loaded from the PDF.")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
split_documents = text_splitter.split_documents(documents)

# Use HuggingFaceEmbeddings instead of SentenceTransformerEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a FAISS vector store
vectorstore = FAISS.from_documents(split_documents, embeddings)

# Save the vector store
vectorstore.save_local("faiss_index")

# Load the vector store with `allow_dangerous_deserialization=True`
new_vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)


generator = pipeline("text2text-generation", model="google/flan-t5-base", device='cpu')
llm = HuggingFacePipeline(pipeline=generator)

# Create a RetrievalQA chain
retriever = new_vectorstore.as_retriever()
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# Query the chain
query = "Give me developmet goals"
response = retrieval_chain.run(query)

print(response)

