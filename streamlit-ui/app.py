import streamlit as st
import requests
from typing import Dict, Any
import os
import json

# Page config
st.set_page_config(
    page_title="SmartDoc Research Agent",
    page_icon="🔬",
    layout="wide"
)

# API Configuration
API_BASE_URL = os.getenv("AGENT_API_URL", "http://localhost:8002")

def main():
    st.title("🔬 SmartDoc Research Agent")
    st.markdown("Intelligent research assistant powered by AI")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "current_topic" not in st.session_state:
        st.session_state.current_topic = None

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Check API health
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.error("❌ API Error")
        except:
            st.error("❌ API Unavailable")
            st.info(f"Trying to connect to: {API_BASE_URL}")
            
        if st.session_state.session_id:
            st.success(f"Active Session: {st.session_state.session_id[:8]}...")
            if st.button("End Session"):
                st.session_state.session_id = None
                st.session_state.current_topic = None
                st.session_state.messages = []
                st.rerun()
    
    # Main interface
    if not st.session_state.session_id:
        st.header("Start Research")
        topic = st.text_input("Research Topic", placeholder="Enter your research topic...")
        
        col1, col2 = st.columns(2)
        with col1:
             depth = st.selectbox("Research Depth", ["basic", "intermediate", "advanced"], index=1)
        
        if st.button("Start Research Session"):
            if topic:
                with st.spinner("Initializing Research Agent..."):
                    try:
                        payload = {
                            "topic": topic,
                            "research_depth": depth,
                            "optimization_level": "balanced",
                            "enable_streaming": False
                        }
                        response = requests.post(f"{API_BASE_URL}/research/session", json=payload)
                        
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.session_id = data["session_id"]
                            st.session_state.current_topic = topic
                            st.success(f"Research session started for: {topic}")
                            st.rerun()
                        else:
                            st.error(f"Error creating session: {response.text}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
            else:
                st.error("Please enter a research topic")
    
    else:
        # Active session interface
        st.info(f"🔬 Researching: **{st.session_state.current_topic}**")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your research..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Send to agent API
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("🤔 Thinking...")
                
                try:
                    payload = {
                        "message": prompt,
                        "stream": False,
                        "depth": "intermediate"
                    }
                    response = requests.post(
                        f"{API_BASE_URL}/research/chat/{st.session_state.session_id}", 
                        json=payload,
                        timeout=120 # Extended timeout for research
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        # Extract answer from result (adjust key based on API response structure)
                        answer = result.get("response", result.get("answer", str(result)))
                        
                        message_placeholder.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # Show sources if available
                        if "sources" in result and result["sources"]:
                            with st.expander("📚 Sources"):
                                for source in result["sources"]:
                                    st.markdown(f"- [{source.get('title', 'Link')}]({source.get('url', '#')})")
                    else:
                        error_msg = f"Error: {response.text}"
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        
                except Exception as e:
                    error_msg = f"Connection error: {str(e)}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
