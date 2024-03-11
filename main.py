import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import os

def init():
  load_dotenv()

  # Page Title
  st.set_page_config(page_title="K.I.M.I Chatbot with Document", page_icon="🤖")

def main():
  init()

  # Header
  st.markdown('<p style="text-align: center; font-size: 32px; font-weight: 600; margin-bottom: 10px;">K.I.M.I</p>', unsafe_allow_html=True)
  st.markdown('<p style="text-align: center; font-size: 24px; font-weight: 500; margin-bottom: 30px;">Ask me anything with your document</p>', unsafe_allow_html=True)

  # Container content
  container = st.container(border=True)

  with st.sidebar:
    # Upload file/document
    file_upload = st.file_uploader('Upload your document', type=['pdf', 'csv']) 

  if file_upload is not None:
    # Read the file/document
    file_reader = PdfReader(file_upload)

    # Extract all text of the file into raw_text 
    raw_text = ''
    for content in file_reader.pages:
      raw_text += content.extract_text()

    # Split it into chunks
    text_split = CharacterTextSplitter(
      separator="\n", 
      chunk_size=1000, 
      chunk_overlap=200, 
      length_function=len
    )
    chunks = text_split.split_text(raw_text)
    
    # Embeddings instance
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Convert it into embeddings using FAISS
    create_embeddings = FAISS.from_texts(chunks, embeddings)

  # User prompt
  if prompt := st.chat_input('Send a message to K.I.M.I'):
    # Similarity search
    docs = create_embeddings.similarity_search(prompt)

    # Process Question answering
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type='stuff')
    response = chain.run(input_documents=docs, question=prompt)

    # Output message
    container.chat_message('human').write(prompt)
    container.chat_message('assistant').write(f"K.I.M.I : {response}")


if __name__ == '__main__':
  main()