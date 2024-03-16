import streamlit as st
from httpx_oauth.clients.google import GoogleOAuth2

# Sign in with google initialize
CLIENT_ID = st.secrets['CLIENT_ID']
CLIENT_SECRET = st.secrets['CLIENT_SECRET']
REDIRECT_URI = 'http://localhost:8501/'

client = GoogleOAuth2(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)

async def get_authorization_url(_client: GoogleOAuth2, redirect_uri: str):
    return await _client.get_authorization_url(redirect_uri, scope=["profile", "email"])

async def get_access_token(client: GoogleOAuth2, redirect_uri: str, code: str):
    return await client.get_access_token(code, redirect_uri)

async def get_email(client: GoogleOAuth2, token: str):
    user_id, user_email = await client.get_id_email(token)
    return user_id, user_email
