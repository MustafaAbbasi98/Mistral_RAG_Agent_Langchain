# Mistral_RAG_Agent_Langchain

## Description
A streamlit app allowing users to chat with their PDF documents and use other tools with a Mistral Agent created using Langchain.

The model currently has access to the following tools:
- Wikipedia
- Calculator
- Research (allows you to query your PDF)

## Usage
1. Clone the repository.
2. Set up and activate a python virtual enviornment (optional).
3. Run `pip install requirements.txt` to install necessary dependencies.
4. Run `streamlit run app.py --server.enableXsrfProtection false` to launch the streamlit app. Note that you must disable CORS to successfully upload your PDF file.
5. Follow instructions given in the app. You will need to setup your `Huggingface Hub` API key and upload a valid PDF document before you can query the LLM.
6. Read the different tips given in the app to improve your queries.

## How it works

### Agent
- When the user submits a query, the underlying `Mistral-Instruct-v0.3` LLM agent decides which tool to use to generate a response.
- The LLM decides the tool and calls the tool with the correct arguments.
- The output of this tool is fed back into the LLM which helps it return it's final answer.
- Note: I found that version 0.3 of `Mistral-Instruct` performs better than version 0.2 for tool calling. 
- I also had to use custom prompt engineering to improve Mistral's tool calling. Through experiments, I found out that a `react-json` agent that outputs JSON works better than the typical `react`-only agent.

### Retrieval Augmented Generation (RAG)
- If the LLM decides to use the Research tool, it will end up invoking the custom RAG chain created on top of the original PDF document uploaded by the user.
- The custom RAG chain uses `PDFMiner` to load the documents, a splitter to split documents into chunks, a Huggingface `BGE` model to embed those chunks, and a `Chroma` retriever. 
- I also had to use prompt engineering to develop a custom prompt that would allow Mistral to perform RAG well.

## Helpful References
I found the following references and tutorials to be very helpful when developing this application:
- https://medium.com/@jorgepardoserrano/building-a-langchain-agent-with-a-self-hosted-mistral-7b-a-step-by-step-guide-85eda2fbf6c2
- https://medium.com/@thakermadhav/build-your-own-rag-with-mistral-7b-and-langchain-97d0c92fa146
- https://medium.com/@thakermadhav/part-2-build-a-conversational-rag-with-langchain-and-mistral-7b-6a4ebe497185
- https://realpython.com/build-llm-rag-chatbot-with-langchain/

