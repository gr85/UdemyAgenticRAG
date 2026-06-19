from dotenv import load_dotenv, find_dotenv

from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic


load_dotenv(find_dotenv(".env"))

llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
client = Client()
prompt = client.pull_prompt("rlm-eu/rag-prompt", dangerously_pull_public_prompt=True)
generation_chain = prompt | llm | StrOutputParser()