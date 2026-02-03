import streamlit as st
from utils.pdf_processor import PDFProcessor
from utils.vector_store import VectorStoreManager
from utils.llm_handler import LLMHandler
import os
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path(__file__).parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

api_key = os.getenv("API_KEY")
if api_key:
    api_key = api_key.strip()

# Page config
st.set_page_config(
    page_title="Research Paper Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Research Paper RAG Analyzer")
st.markdown("Upload a research paper and ask questions about it!")


# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# File upload
uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type=['pdf'])

if uploaded_file and api_key:
    if not st.session_state.processed:
        with st.spinner("Processing paper..."):
            # Process PDF
            processor = PDFProcessor()
            chunks = processor.process_pdf(uploaded_file)

            # Create vector store
            vector_manager = VectorStoreManager()
            vectorstore = vector_manager.create_vectorstore(chunks)

            # Setup LLM
            llm_handler = LLMHandler(api_key)
            st.session_state.qa_chain = llm_handler.create_qa_chain(vectorstore)
            st.session_state.processed = True

            st.success(f"✅ Processed {len(chunks)} chunks from the paper!")

    st.subheader("💬 Ask Questions")

    question = st.text_input(
        "Your Question:",
        value=st.session_state.get('current_question', ''),
        key='question_input'
    )

    if question and st.button("Get Answer", type="primary"):
        with st.spinner("Analyzing..."):
            result = st.session_state.qa_chain({"query": question})

            st.markdown("### 📝 Answer")
            st.write(result['result'])

            with st.expander("📚 View Source Passages"):
                for i, doc in enumerate(result['source_documents']):
                    st.markdown(f"**Source {i + 1} (Page {doc.metadata.get('page', 'N/A')})**")
                    st.text(doc.page_content[:300] + "...")
                    st.divider()