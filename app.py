import streamlit as st
import requests
import json
from io import BytesIO
from PIL import Image

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Document Q&A System", layout="wide")

st.title("📄 Document Question Answering System")

# Initialize session state
if 'session_id' not
