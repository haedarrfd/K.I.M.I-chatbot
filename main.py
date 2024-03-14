import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import firebase_admin
from firebase_admin import credentials, firestore
import time
from datetime import datetime
from uuid import uuid4
import os

# Get keys from .env
load_dotenv()

# Page Title
st.set_page_config(page_title="K.I.M.I Chatbot with Document", page_icon="🤖", layout='wide')

if not firebase_admin._apps:
  # Firebase credentials
  cred = credentials.Certificate("key.json")
  firebase_admin.initialize_app(cred)

# Firestore instance
db = firestore.client()

# Handle upload file
def uploadFile():
  with st.sidebar:
    # Upload file/document
    file_uploader = st.file_uploader('Upload your document', help='PDF or CSV only', type=['pdf', 'csv'])

    # Check if file isn't uploaded
    if file_uploader is None:
      st.warning('Please upload a file before continue!')

    for _space in range(6):
      st.write("\n")

    # Footer
    st.markdown('<p style="text-align: center; font-size: 13px; font-weight: 400; opacity: 0.8; margin-top: auto;">K.I.M.I may provide inaccurate information, Consider checking important information.</p>', unsafe_allow_html=True)

    return file_uploader

# Handle process file pdf
def handleFilePDF(file):
  # Read the file/document PDF
  file_reader = PdfReader(file)

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

  return create_embeddings

# Handle process file csv
def handleFileCSV(llm, file):
  # Read the file/document CSV
  df = pd.read_csv(file)

  # Construct pandas agent
  agent = create_pandas_dataframe_agent(llm, df, verbose=True)

  return agent    

def main():
  try:
    # Header
    st.markdown('<p style="text-align: center; font-size: 32px; font-weight: 600; margin-bottom: 5px;">K.I.M.I</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 22px; font-weight: 500; margin-bottom: 30px;">Ask me anything with your document</p>', unsafe_allow_html=True)

    # Container content
    container = st.container(border=True)

    # OpenAI instance
    llm = OpenAI(temperature=0)

    # Instance upload file
    upload_file = uploadFile()

    if upload_file is not None:
      # Handle PDF file
      if upload_file.type == 'application/pdf':
        data_pdf = handleFilePDF(upload_file)

      # Handle CSV file
      elif upload_file.type == 'text/csv':
        data_csv = handleFileCSV(llm, upload_file)        

    # Timestamps
    timestamp = time.time()
    # convert to datetime
    date_time = datetime.fromtimestamp(timestamp)
    # convert timestamp to string
    str_date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")
    
    # User prompt
    prompt = st.chat_input('Send a message to K.I.M.I')

    # Handle loading
    with st.spinner('Loading...'):
      if prompt != '' and prompt is not None:
        # Handle prompt based on PDF file 
        if upload_file.type == 'application/pdf':
          # Similarity search
          docs = data_pdf.similarity_search(prompt)

          # Process Question answering
          chain = load_qa_chain(llm, chain_type='stuff')
          response_pdf = chain.run(input_documents=docs, question=prompt)

          # Create the result to firestore
          # db.collection('chat_history').document().set({
          #   'history_id': str(uuid4()),
          #   'user_input': prompt,
          #   'bot_response': response_pdf,
          #   'timestamps': str_date_time
          # })

          # Output message
          container.chat_message('human').write(prompt)
          container.chat_message('assistant').write(f"K.I.M.I : {response_pdf}")

        # Handle prompt based on CSV file 
        elif upload_file.type == 'text/csv':
          # Agent process to thinking
          response_csv = data_csv.run(prompt)

          # Output message
          container.chat_message('human').write(prompt)
          container.chat_message('assistant').write(f"K.I.M.I : {response_csv}")

        else:
          st.error('Something wrong, try again!')

    

  except Exception as err:
    st.error(f"Something wrong, Please be sure upload a file or the file format! {err}")


if __name__ == '__main__':
  main()