import asyncio

from dotenv import load_dotenv, find_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv(find_dotenv(".env"))




retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OllamaEmbeddings(model="qwen3-embedding:4b"),
).as_retriever()



async def main():
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
    ]
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun_many(urls)
        docs_list = [Document(
            page_content=doc.markdown,
            metadata={"url": doc.url}
        ) for doc in result]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(docs_list)
    vector_store = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model="qwen3-embedding:4b"),
        persist_directory="./.chroma",
    )





if __name__ == "__main__":
    pass
    # asyncio.run(main())

