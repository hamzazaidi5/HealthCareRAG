import streamlit as st
from utils.config import Config
from utils.data_loader import OncologyDataLoader, load_csv_as_documents
from chain.custom_chain import DrugRecommendationChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


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

    # Return retriever for custom chain creation
    return retriever


# Create a conversational chain for follow-up questions
def create_conversational_chain(retriever):
    llm = ChatGroq(temperature=0, model_name=Config.LLM_MODEL)

    prompt = ChatPromptTemplate.from_template(
        """You are an oncology expert. Answer the following question about cancer drugs concisely.

        Context: {context}

        Question: {question}

        Provide a direct and brief response without the structured format. Just answer in 2-3 sentences."""
    )

    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain


# Use Streamlit caching to load the system only once
@st.cache_resource
def load_system():
    retriever = initialize_system()
    initial_chain = DrugRecommendationChain(retriever)
    followup_chain = create_conversational_chain(retriever)
    return initial_chain, followup_chain


def main():
    st.title("Oncology Drug Recommendation Chatbot")

    # Initialize session state for chat history if not already done
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "context" not in st.session_state:
        st.session_state.context = {}

    if "is_first_query" not in st.session_state:
        st.session_state.is_first_query = True

    # Load your recommendation systems
    initial_chain, followup_chain = load_system()

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**User:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")

    # Get user query
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask about oncology drugs:", key="user_message")
        submit_button = st.form_submit_button("Send")

    if submit_button and user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Determine which chain to use based on whether this is the first query
        if st.session_state.is_first_query:
            # Use the structured template for the first query
            response = initial_chain.invoke(user_input)
            st.session_state.is_first_query = False

            # Extract the drug name from the response if available
            if "✅ Drug Name:" in response:
                drug_line = [line for line in response.split('\n') if "✅ Drug Name:" in line][0]
                drug_name = drug_line.split("✅ Drug Name:")[1].strip()
                if drug_name != "Drug is not available":
                    st.session_state.context["current_drug"] = drug_name
        else:
            # Use the conversational chain for follow-up questions
            enhanced_query = user_input
            if "current_drug" in st.session_state.context:
                enhanced_query = f"Regarding {st.session_state.context['current_drug']}: {user_input}"

            response = followup_chain.invoke(enhanced_query)

        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Rerun to update the UI
        st.rerun()

    # Add a clear button to reset the conversation
    if st.button("Start New Conversation"):
        st.session_state.chat_history = []
        st.session_state.context = {}
        st.session_state.is_first_query = True
        st.rerun()

    # Add a button to explicitly ask about a new drug (resets to formal prompt)
    if st.button("Ask About a New Drug"):
        st.session_state.is_first_query = True
        st.session_state.context = {}
        st.rerun()


if __name__ == "__main__":
    main()