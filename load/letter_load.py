import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredFileLoader  # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings # for creating embeddings
from langchain.vectorstores import  Pinecone # for the vectorization part
from langchain.text_splitter import TokenTextSplitter
import pinecone

#import magic
#import nltk
#nltk.download('punkt')

# identify the various pdf files
pdfs = [file for file in os.listdir('./letters/') if 'pdf' in file]

# loops through each pdf in the letters directory
# and loops the content using langchains PyPDFLoader
# there can likely be a better way for loading each individual doc
# but I ran into issues with some of the other loader dependencies
# ideally each letter would be serialized itself
# but this approach just consolidates and loads them all in a flat list
page_list = []
for pdf in pdfs:
    pdf_path = f"./letters/{pdf}"
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    page_list.append(pages)

flat_list = [item for sublist in page_list for item in sublist]

# initialize pinecone
import streamlit as st
pinecone.init(
    api_key=st.secrets['pinecone_key'], 
    environment=st.secrets['pinecone_env'] 
)
index_name = "buffett"

text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(flat_list)

# note you should create an OPENAI_API_KEY env environment variable or use st.secrets
# create embeddings using OpenAI and load into Pinecone 
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets['openai_key'])
docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
