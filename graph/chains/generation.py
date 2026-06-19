from dotenv import load_dotenv, find_dotenv

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic


llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)