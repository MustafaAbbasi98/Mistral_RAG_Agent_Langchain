import os
import streamlit as st
from PyPDF2 import PdfFileReader
from rag_chain import load_chain
from agent import create_agent
import tempfile

def get_pdf_title(pdf_file_path):
    pdf_reader = PdfFileReader(open(pdf_file_path, "rb")) 
    return pdf_reader.getDocumentInfo().title

def process_file():
    st.session_state["chain"] = None
    st.session_state["agent"] = None
    
    file = st.session_state["file_uploader"]
    if not file or not hf_api_key:
        return
    
    #Save as temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name
    
    #Ingest file and create the RAG chain followed by the Agent
    with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["chain"] = load_chain(file_path)
            st.session_state["agent"] = create_agent(st.session_state.chain)
    
    #Remove temp file  
    os.remove(file_path)

st.title("🦜🔗 Mistral RAG Agent Demo")

hf_api_key = st.sidebar.text_input("Hugginface Hub API Key", type="password")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key
st.session_state.hf_api_key = hf_api_key

uploaded_file = st.sidebar.file_uploader('Choose your .pdf file', 
                                         type="pdf",
                                         key="file_uploader",
                                         on_change=process_file, #run on every file change
                                         )

st.session_state["ingestion_spinner"] = st.empty()

# st.sidebar.button("Submit File", on_click=process_file)

def generate_response(query):
    try:
        response = st.session_state.agent.invoke({"input": query})['output']
        response = response.strip().strip('</s>')
    except Exception as e:
        print(e)
        response = "Oops! There seems to be an issue. Try again. Maybe try changing the prompt."
    
    return response

with st.form("my_form"):
    query = st.text_area(
        "Enter query:",
        "Who is Elon Musk?",
    )
    submitted = st.form_submit_button("Submit")
    
    if not hf_api_key.startswith("hf_"):
        st.warning("Please enter your valid HuggingfaceHub API key!", icon="⚠")
        
    if uploaded_file is None:
        st.warning("Please upload a valid pdf file!", icon="⚠")
           
    # if not st.session_state.get("chain") or not st.session_state.chain:
    #     st.warning("Please submit your file!", icon="⚠")
        
    if submitted and uploaded_file and st.session_state.get("chain") and hf_api_key.startswith("hf_"):
        # path = print(uploaded_file.name)
        with st.spinner("Generating..."):
            response = generate_response(query)
            st.info("Assistant: \n\n" + response)
            
with st.expander("View Tips and Instructions"):
    
    st.markdown("""
                Instructions:
                
                • Enter your huggingface hub api key.
                
                • Upload your PDF (research) document you want to chat with. This will take some time depending on the size of your PDF.
                
                • Type your query and click "Submit" to obtain a response from the LLM.
                
                Tips:
                
                • The LLM has access to wikipedia, a calculator and your PDF document.
                
                • If the LLM fails to respond, guide it by telling it what tool to use. Use a phrase like "You MUST do research"
                
                • Try to frame your queries as questions.
                
                • Avoid using punctuation marks at end of sentences. For some reason, it can mess up the LLM.
                
                • Avoid queries that require multi-step reasoning or multi-step tool usage.        
                
                • If you still want multi-step/tool reasoning try appending the following phrase at the end of your query:
                "You MUST think step-by-step"
                
                """
                )

        
        