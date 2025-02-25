# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Optional
# import pandas as pd
# from langchain_groq import ChatGroq
# from langchain.schema import HumanMessage, SystemMessage
# from dotenv import load_dotenv
# import os
#
# load_dotenv()
#
# app = FastAPI()
#
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )
#
# # Load and process the CSV data
# df = pd.read_csv('data/drugs.csv')
# context = df.to_string()
#
# # Initialize Groq
# llm = ChatGroq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     model_name="mixtral-8x7b-32768"
# )
#
#
# class Query(BaseModel):
#     question: str
#     user_conditions: Optional[str] = None
#
#
# @app.post("/query")
# async def query_drugs(query: Query):
#     try:
#         system_prompt = f"""You are a medical advisor AI. Use the following drug database information to answer questions:
#         {context}
#
#         Provide clear, accurate responses about drug suitability based on the database.
#         If you're unsure or if the information isn't in the database, say so explicitly.
#         Always consider user conditions when making recommendations."""
#
#         messages = [
#             SystemMessage(content=system_prompt),
#             HumanMessage(content=f"User conditions: {query.user_conditions}\nQuestion: {query.question}")
#         ]
#
#         response = llm.invoke(messages)
#
#         return {"response": response.content}
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# @app.get("/health")
# async def health_check():
#     return {"status": "healthy"}


from utils.config import Config
from utils.data_loader import OncologyDataLoader, load_csv_as_documents
from chain.custom_chain import DrugRecommendationChain
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def initialize_system():
    # Load data
    loader = OncologyDataLoader(Config.CSV_PATH)
    documents = load_csv_as_documents(Config.CSV_PATH)

    # Create embeddings
    # embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    embeddings = OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        openai_api_key=Config.OPENAI_API_KEY
    )

    # Create vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Create chain
    return DrugRecommendationChain(retriever)


if __name__ == "__main__":
    system = initialize_system()

    print("Oncology Drug Recommendation System")
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        response = system.invoke(query)
        print("\nRecommendation:")
        print(response)