# Created By: Karanpreet Sachdeva
# Description: A simple chat bot agent that is in console level yet power ful and take the data from wikipedia
# Importing necessary packages
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()
# Setting up a LLM
llm2 = ChatAnthropic(model="claude-2-sonnet-20241022")
llm3 = ChatGroq(model="llama-3.3-70b-versatile")

# Test should result the content of the capital of India which is New Delhi
# response = llm3.invoke("What is the capital of India?")
# print(response.content)

# Specifying all the fields needed from the output of the LLM
class ReaserchResponse(BaseModel):
    topic: str
    summary: str
    source: list[str]
    tools_used: list[str]
# Setting up the output parser
parser = PydanticOutputParser(pydantic_object=ReaserchResponse)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful research assistant that can answer questions and provide summaries on various topics. "
            "You will be given a topic to research, and you should provide a concise summary of the topic, "
            "along with a list of sources you used for your research and the tools you utilized to gather information.\n{format_instructions}""",
        ),
        ("placeholder","{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())