import streamlit as st
from utils.config import Config
from utils.data_loader import OncologyDataLoader, load_csv_as_documents
from chain.custom_chain import DrugRecommendationChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def initialize_system():
    documents = load_csv_as_documents(Config.CSV_PATH)

    # Create embeddings
    embeddings = OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        openai_api_key=Config.OPENAI_API_KEY
    )

    # Create vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Create chain
    return DrugRecommendationChain(retriever)

# Use Streamlit caching to load the system only once.
@st.cache_resource
def load_system():
    return initialize_system()

def main():
    st.title("Oncology Drug Recommendation System")

    # Load your recommendation system
    system = load_system()

    # Get user query
    query = st.text_input("Enter your question:")

    if st.button("Get Recommendation"):
        if query:
            # Invoke your chain on the user's query
            response = system.invoke(query)
            st.subheader("Recommendation:")
            st.write(response)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
