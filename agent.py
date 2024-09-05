from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import WikipediaQueryRun, Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain_huggingface import HuggingFaceEndpoint
import numexpr as ne
from typing import Union

react_json_prompt_template = """Answer the following questions as best you can.
You can answer directly if the user is greeting you or similar.
Otherise, you have access to the following tools:

{tools}

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).
The only values that should be in the "action" field are: {tool_names} (Can not be None and must be a valid tool)
The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions.
Here is an example of a valid $JSON_BLOB:
```
{{{{
    "action": $TOOL_NAME,
    "action_input": $INPUT
}}}}
```

Here is an example of an invalid $JSON_BLOB:
```
{{{{
    "action": none,
    "action_input": $INPUT
}}}}
```

Again, DO NOT return a list of MULTIPLE actions.

The $JSON_BLOB must always be enclosed with triple backticks!

ALWAYS use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action:
```
$JSON_BLOB
```
Observation: the result of the action...
(this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Use this format if you want to respond directly to the human OR when you have the final answer:

Thought: I now have the final answer.
Final Answer: (fill answer here)

Again do not forget, you must use the format as described above.

Begin! Reminder to always use the exact characters `Final Answer` when responding.'

Question: {input}
Thought: {agent_scratchpad}
"""


@tool
def calculator(expression: str) -> Union[str, int, float]:
  """"Use this tool for math operations. You MUST MUST MUST use correct numexpr syntax. Use it always you need to solve any math operation. Be sure syntax is correct."""
  print("\n", expression, len(expression), type(expression))
  try:
    return ne.evaluate(expression).item()
  except Exception:
    return "This is not a numexpr valid syntax. Try a different syntax."

#We will define our tools over here
def create_tools(rag_chain):
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2500))
    wikipedia_tool = Tool(
        name="wikipedia",
        description="Use when you need to search for information that you do not know about. NEVER EVER search for more than one concept at a single step. If you need to compare two concepts, search for each one individually one by one. Syntax: string with a simple concept",
        func=wikipedia.run
        )
    
    #To convert our RAG chain to a tool, we can use the Tool constructor and pass our chain's invoke function
    rag_tool = Tool(name="research",
                    func=rag_chain.invoke,
                    description="""Useful when you need to answer questions
                    about research related topics, specific research papers, pdf documents, or any other qualitative
                    question that could be answered using semantic search. Not useful for answering objective questions
                    that involve counting, percentages, aggregations, or listing facts. Use the
                    entire prompt as input to the tool. For instance, if the prompt is
                    "What is the title of this paper?", the input should be
                    "What is the title of this paper?". Only use when other tools fail.
                    Also, do NOT summarize the results from this tool when you give your final response.
                    """)
    
    tools = [wikipedia_tool, calculator, rag_tool]
    
    return tools


def create_agent(rag_chain):
    
    tools = create_tools(rag_chain)
    
    #You can try a different LLM over here (OpenAI, LLaMa, etc.)
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        max_new_tokens=5000,
        do_sample=False,
        temperature=0.3,
        repetition_penalty=1.03)
    
    react_json_prompt = PromptTemplate.from_template(template=react_json_prompt_template)
    
    #We create a custom React Json Agent
    agent = create_react_agent(llm=llm,
                           tools=tools,
                           prompt=react_json_prompt,
                           output_parser=ReActJsonSingleInputOutputParser())
    
    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent,
                                tools=tools,
                                #  memory=memory,
                                # verbose=True,
                                # return_intermediate_steps=True,
                                handle_parsing_errors=True)
    
    return agent_executor
    