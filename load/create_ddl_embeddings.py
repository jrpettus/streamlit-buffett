from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import streamlit as st

# load the ddl file
loader = TextLoader('load/ddls.sql')
data = loader.load()

# split the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
texts = text_splitter.split_documents(data)

# created embeddings from the doc
embeddings = OpenAIEmbeddings(openai_api_key = st.secrets["openai_key"])
docsearch = FAISS.from_documents(texts, embeddings)

# save the faiss index
docsearch.save_local("faiss_index")
