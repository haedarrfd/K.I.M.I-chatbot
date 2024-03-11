import streamlit as st
from dotenv import load_dotenv
import os

def init():
  load_dotenv()

  # Page Title
  st.set_page_config(page_title="K.I.M.I Chatbot with Document", page_icon="🤖")

def main():
  init()

  # Header
  st.markdown('<p style="text-align: center; font-size: 40px; font-weight: 600; margin-bottom: 10px;">K.I.M.I</p>', unsafe_allow_html=True)
  st.markdown('<p style="text-align: center; font-size: 30px; font-weight: 500; margin-bottom: 50px;">Ask me anything with your document</p>', unsafe_allow_html=True)

  with st.sidebar:
    file_upload = st.file_uploader('Upload your document', type=['pdf', 'csv'])
    if file_upload is not None:
      st.write(file_upload.type == 'application/pdf')



if __name__ == '__main__':
  main()