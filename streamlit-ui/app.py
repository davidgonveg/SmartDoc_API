"""
SmartDoc Research Agent - Streamlit UI
"""

import streamlit as st
import requests
from typing import Dict, Any
import os

# Page config
st.set_page_config(
    page_title="SmartDoc Research Agent",
    page_icon="🔬",
    layout="wide"
)

# API Configuration
API_BASE_URL = os.getenv("AGENT_API_URL", "http://localhost:8001")

def main():
    st.title("🔬 SmartDoc Research Agent")
    st.markdown("Intelligent research assistant powered by AI")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Check API health
        try:
            response = requests.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.error("❌ API Error")
        except:
            st.error("❌ API Unavailable")
    
    # Main interface
    st.header("Start Research")
    
    topic = st.text_input("Research Topic", placeholder="Enter your research topic...")
    
    if st.button("Start Research Session"):
        if topic:
            # TODO: Create research session
            st.success(f"Research session started for: {topic}")
        else:
            st.error("Please enter a research topic")
    
    # Chat interface (placeholder)
    st.header("Chat with Agent")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your research..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # TODO: Send to agent API
        with st.chat_message("assistant"):
            response = f"Echo: {prompt}"
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
