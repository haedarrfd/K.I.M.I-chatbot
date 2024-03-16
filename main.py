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
from httpx_oauth.clients.google import GoogleOAuth2
import asyncio
from Oauth import *
import streamlit.components.v1 as components

# Get keys from .env
load_dotenv()

# Page Title
st.set_page_config(page_title="K.I.M.I Chatbot with Document", page_icon="🤖", layout='wide')

# Initialize Firebase app
if not firebase_admin._apps:
  cred = credentials.Certificate("key.json")
  firebase_admin.initialize_app(cred)

# Firestore instance
firebase_firestore = firestore.client()

# Sign in with google initialize
CLIENT_ID = st.secrets['CLIENT_ID']
CLIENT_SECRET = st.secrets['CLIENT_SECRET']
REDIRECT_URI = 'http://localhost:8501/'

# Timestamps
timestamp = time.time()
# convert to datetime
date_time = datetime.fromtimestamp(timestamp)
# convert timestamp to string
str_date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")

# Handle upload file
def uploadFile():
  with st.sidebar:
    # Upload file/document
    file_uploader = st.file_uploader('Upload your document', help='PDF or CSV only', type=['pdf', 'csv'])

    # Check if file isn't uploaded
    if file_uploader is None:
      st.warning('Please upload a file before continue!')

    # Sign out button
    if st.button('Sign Out', type='primary', key='logout'):
      st.session_state.token = None
      st.rerun()

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

# Add data user to firestore
def dataUser(id, email, timestamp):
  # If user already exists in the collection firestore don't add new user
  docs = firebase_firestore.collection('users').where('email', '==', email).stream()
  isUserExists = any(doc.exists for doc in docs)

  if id is not None and email is not None and timestamp is not None:
    if not isUserExists:
      data = firebase_firestore.collection('users').document().set({
        'id': str(id),
        'email': str(email),
        'timestamp': str(timestamp)
      })

  return 

def home(user_email='', user_id=''):
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

          # Store the result to firestore
          # firebase_firestore.collection('chat_histories').document().set({
          #   'history_id': str(uuid4()),
          #   'file_type': upload_file.type,
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

          # Store the result to firestore
          # firebase_firestore.collection('chat_histories').document().set({
          #   'history_id': str(uuid4()),
          #   'file_type': upload_file.type,
          #   'user_input': prompt,
          #   'bot_response': response_csv,
          #   'timestamps': str_date_time
          # })

          # Output message
          container.chat_message('human').write(prompt)
          container.chat_message('assistant').write(f"K.I.M.I : {response_csv}")

        else:
          st.error('Something wrong, try again!')
          
  except Exception as err:
    st.error(f"Something wrong, Please be sure upload a file or the file format! {err}")

# Sign In page
def signInPage(url = ''):
  components.html(f""" 
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>

    <style>
      body {{
        background-color: #1e293b;
      }}
      .btn-primary {{
        background-color: #1d4ed8;
        border: 1px solid #1e3a8a;
        padding: 10px 12px;
        font-size: 16px;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8px;
      }}                   
      .google_icon {{
        width: 20px;
        height: auto;
        text-align: center;
        display: flex;
        align-items: center;
      }}
      </style>
                  
      <div class="container mx-auto text-center">
        <div class="row justify-content-center">
          <div class="col-md-3 text-center mt-5">
            <a href="{url}" target="_blank" class="btn btn-primary">
              Sign In with google
              <div class="google_icon">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 488 512"><path fill="#FFD43B" d="M488 261.8C488 403.3 391.1 504 248 504 110.8 504 0 393.2 0 256S110.8 8 248 8c66.8 0 123 24.5 166.3 64.9l-67.5 64.9C258.5 52.6 94.3 116.6 94.3 256c0 86.5 69.1 156.6 153.7 156.6 98.2 0 135-70.4 140.8-106.9H248v-85.3h236.1c2.3 12.7 3.9 24.9 3.9 41.4z"/></svg> 
              </div>
            </a>
          </div>
        </div>
      </div>
""", height=200)

def main():
  client: GoogleOAuth2 = GoogleOAuth2(CLIENT_ID, CLIENT_SECRET)
 
  if 'user_id' not in st.session_state:
    st.session_state.user_id = ''

  if 'user_email' not in st.session_state:
    st.session_state.user_email = ''
  
  if 'token' not in st.session_state:
    st.session_state.token = None

  if st.session_state.token is None:
    # Get auth url
    authorization_url = asyncio.run(
      get_authorization_url(_client=client, redirect_uri=REDIRECT_URI))
    try:
      code = st.query_params.get('code')
    except:
      signInPage(authorization_url)
    else:
      try:
        token = asyncio.run(get_access_token(client=client, redirect_uri=REDIRECT_URI, code=code))
      except:
        signInPage(authorization_url)
      else:
        if token.is_expired():
          if token.is_expired():
            signInPage(authorization_url)
            st.warning('Login session has ended, Please Sign in again!')
        else:
          # Store the token to token session 
          st.session_state.token = token
          # Run get email and grab the id and email
          user_id, user_email = asyncio.run(get_email(client=client, token=token['access_token']))
          st.session_state.user_id = user_id      
          st.session_state.user_email = user_email
          home(user_id=st.session_state.user_id, user_email=st.session_state.user_email)
          # Clear query params
          st.query_params.clear()
          # Store user to the firestore collection
          if user_id is not None and user_email is not None:
            dataUser(id=st.session_state.user_id, email=st.session_state.user_email, timestamp=str_date_time)
  else:
    home(user_id=st.session_state.user_id, user_email=st.session_state.user_email)


if __name__ == '__main__':
  main()