import streamlit as st
import os
from dotenv import load_dotenv
from ingest import DocumentIngestor
from vector_store import VectorStore
from llm import LLMInterface
from constants import URL

load_dotenv()


st.set_page_config(
    page_title="AngelOne Support Chatbot",
    page_icon="🤖",
    layout="wide"
)

def initialize_system():
    vector_store = VectorStore()
    
    if vector_store.load("vector_store.pkl"):
        st.success("✅ Loaded existing knowledge base")
        with st.spinner("Loading AI model..."):
            try:
                llm = LLMInterface(model_type="flan-t5")
                st.success("✅ AI model loaded successfully")
            except Exception as e:
                st.error(f"❌ Error loading AI model: {e}")
                llm = LLMInterface(model_type="fallback")
        return vector_store, llm
    else:
        st.warning("No existing knowledge base found. Creating new one...")
        
        with st.spinner("Loading documents..."):
            ingestor = DocumentIngestor()
            
            all_file_docs = ingestor.load_all_documents()
            st.info(f"Loaded {len(all_file_docs)} documents")
            
            web_docs = ingestor.scrape_website_recursive(
                start_url="https://www.angelone.in/support",
                max_pages=10  # Adjust based on your needs
            )
            st.info(f"Scraped {len(web_docs)} web sections")
            
            all_docs = all_file_docs + web_docs
            chunks = ingestor.process_documents(all_docs)
            
            if not chunks:
                st.error("❌ No documents found! Please add files to 'data/pdfs/' folder")
                return None, None
            
            vector_store.create_embeddings(chunks)
            vector_store.save("vector_store.pkl")
            
            st.success(f"✅ Created knowledge base with {len(chunks)} chunks")
        
        
        with st.spinner("Loading AI model..."):
            try:
                llm = LLMInterface(model_type="flan-t5")
                st.success("✅ AI model loaded successfully")
            except Exception as e:
                st.error(f"❌ Error loading AI model: {e}")
                llm = LLMInterface(model_type="fallback")
        
        return vector_store, llm

def main():
    st.title("🤖 AngelOne Support Chatbot")
    st.markdown("Ask me anything about AngelOne services!")
    
    if 'vector_store' not in st.session_state or 'llm' not in st.session_state:
        vector_store, llm = initialize_system() 
        st.session_state.vector_store = vector_store
        st.session_state.llm = llm
    
    if not st.session_state.vector_store or not st.session_state.llm:
        st.error("❌ Failed to initialize system. Please check your setup.")
        return
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question about AngelOne..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                search_results = st.session_state.vector_store.search(prompt, k=3)
                
                if not search_results:
                    response = "I don't Know. Please ask about insurance policies, trading, account opening, or customer support topics."
                else:
                    
                    context = st.session_state.llm.format_context(search_results)
                    response = st.session_state.llm.generate_response(prompt, context)
                
                st.markdown(response)
                
                if search_results:
                    with st.expander("📚 Sources"):
                        for i, (text, score) in enumerate(search_results):
                            st.write(f"**Source {i+1}** (Relevance: {1-score:.2f})")
                            st.write(text[:300] + "..." if len(text) > 300 else text)
                            st.divider()
                else:
                    with st.expander("ℹ️ Search Info"):
                        st.write("No relevant documents found for this query.")
                        st.write("Try asking about:")
                        st.write("• Insurance policies and coverage")
                        st.write("• Account opening and trading")
                        st.write("• Customer support topics")
                        st.write("• Fees and charges")
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    with st.sidebar:
        st.header("📊 System Info")
        
        if st.session_state.vector_store:
            st.metric("Documents", len(st.session_state.vector_store.documents))
        
        st.divider()
        
        if st.button("🔄 Rebuild Knowledge Base"):
            st.session_state.vector_store = None
            st.session_state.llm = None
            if os.path.exists("vector_store.pkl"):
                os.remove("vector_store.pkl")
            st.rerun()
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()