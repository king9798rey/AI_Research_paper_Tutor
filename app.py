import streamlit as st
import requests

# FastAPI Backend URL
FASTAPI_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Research Paper Tutor", page_icon="🎓", layout="wide")

st.title("🎓 AI Research Paper Tutor for Beginners")
st.markdown("Upload any complex AI/ML research paper and ask questions to understand it in simple terms!")

# --- Sidebar: PDF Upload Section ---
with st.sidebar:
    st.header("📄 Upload Paper")
    uploaded_file = st.file_uploader("Choose a research paper PDF", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process Paper"):
            with st.spinner("Parsing paper with LlamaParse & saving to FAISS... (This may take a moment)"):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{FASTAPI_URL}/upload-paper/", files=files)
                    if response.status_code == 200:
                        st.success("Paper successfully processed and ready for Q&A!")
                        st.session_state["paper_uploaded"] = True
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Could not connect to FastAPI backend: {e}")

# --- Main Section: Chat Interface ---
st.header("💬 Ask Your AI Tutor")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask something about the paper (e.g., What is the main contribution?)"):
    # Check if paper is uploaded
    if not st.session_state.get("paper_uploaded", False):
        st.warning("⚠️ Please upload and process a research paper first using the sidebar!")
    else:
        # Add user message to chat history
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Send request to FastAPI backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking like a friendly tutor..."):
                try:
                    payload = {"question": prompt}
                    response = requests.post(f"{FASTAPI_URL}/question", json=payload)
                    
                    if response.status_code == 200:
                        answer = response.json().get("answer")
                        st.markdown(answer)
                        # Add assistant response to chat history
                        st.session_state["messages"].append({"role": "assistant", "content": answer})
                    else:
                        error_msg = response.json().get("detail", "Error generating answer.")
                        st.error(error_msg)
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")