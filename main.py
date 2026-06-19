from dotenv import load_dotenv, find_dotenv

from graph.graphs import app


load_dotenv(find_dotenv(".env"))

def main():
    print("Hello Advanced RAG")
    print(app.invoke(input={"question": "what is agent memory?"}))


if __name__ == "__main__":
    main()
