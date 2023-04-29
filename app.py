import io
import streamlit as st
import os
import glob
import pandas as pd
from pypdf import PdfReader
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
import openai

st.title('Upload PDF in Pinecone & Query Through GPT-3.5-Turbo')
st.markdown("**OpenAI API key**")
key = st.text_input("Paste Your API key here")
st.write(key)
import os
os.environ["OPENAI_API_KEY"] = key
def process_pdf_files(pdf_file):
    # Read the contents of the uploaded file
    pdf_bytes = pdf_file.read()

    # Use BytesIO to create an in-memory binary stream
    with io.BytesIO(pdf_bytes) as pdf_stream:
        reader = PdfReader(pdf_stream)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

    return text

# Create a file uploader and search box
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)


pdf_name = []
for f in uploaded_files:
    #st.write(f.name)
    pdf_name.append(f.name)
    

data_list = []

# Process uploaded files and filter by search query
if uploaded_files:
    for pdf_file in uploaded_files:
        raw_text = process_pdf_files(pdf_file)
        st.write(raw_text[0:100])

        text_splitter = CharacterTextSplitter(        
            separator = "\n",
            chunk_size =2000,
            chunk_overlap  = 200,
            length_function = len,
        )
        texts = text_splitter.split_text(raw_text)
        # Download embeddings from OpenAI
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
        from langchain.chains.question_answering import load_qa_chain
        from langchain.llms import OpenAI
        
        query = st.text_input("What You want to know?")
        if query:
            chain = load_qa_chain(OpenAI(), chain_type="stuff")
        
            docs = docsearch.similarity_search(query)

            st.write(chain.run(input_documents=docs, question=query))
