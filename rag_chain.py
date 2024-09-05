from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate

#Here is a custom RAG prompt template for Mistral that uses the [INST] tokens
prompt_template = """<s> [INST] Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer".
2. Read the context carefully in detail to come up with the answer.
3. If you find the answer, write the answer in a concise way with five sentences maximum. 
4. Give your answer in a paragraph. DO NOT use bullet points or a list. [/INST] </s>

[INST] Context: {context}

Question: {question}

Answer: [/INST]
"""

RAG_PROMPT = PromptTemplate(
 template=prompt_template, input_variables=["context", "question"]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def extract_splits(pdf_path):
    loader = PDFMinerLoader(pdf_path) #change PDF loader as desired
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, add_start_index=True
        )
    all_splits = text_splitter.split_documents(documents)
    return all_splits
    

def load_retriever(pdf_path):
    all_splits = extract_splits(pdf_path)
    
    huggingface_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",  # alternatively use "sentence-transformers/all-MiniLM-l6-v2" for a lighter and faster experience.
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=huggingface_embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever

def load_chain(pdf_path):
    retriever = load_retriever(pdf_path)
    
    #Change LLM as needed
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        max_new_tokens=5000,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.03)
    
    rag_chain = (
        {'context': retriever | format_docs, 'question': RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
    