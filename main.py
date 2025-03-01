import streamlit as st
from utils.config import Config
from chain.custom_chain import DrugRecommendationChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from utils.data_loader import OncologyDataLoader
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Initialize ChatOpenAI instance used for generating questions.
chat_model = ChatOpenAI(
    temperature=0,
    model_name=Config.LLM_MODEL,
    openai_api_key=Config.OPENAI_API_KEY
)

# Define an initial system message that sets the conversation context.
initial_system_msg = SystemMessage(
    content=(
        "You are a friendly, professional oncology treatment assistant. "
        "Your task is to ask the user a series of up to 5 generic questions to collect key patient data "
        "(e.g., demographics, cancer type, stage, subtype, treatment history). "
        "After these interactions, generate a final query summarizing the patient data to identify the best FDA-approved drug(s) "
        "and available survival data."
    )
)

# Initialize conversation messages in session state using LangChain message objects.
if "messages" not in st.session_state:
    st.session_state.messages = [initial_system_msg]
    # Add a welcome message
    welcome_message = AIMessage(
        content="Hello! I'm your oncology treatment assistant. I'll help you find the most appropriate FDA-approved treatment options based on your situation. Let's start with some questions about the patient. What is the patient's age and gender?")
    st.session_state.messages.append(welcome_message)

# Display the conversation history.
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    # Hide system messages from the UI
    # elif isinstance(msg, SystemMessage):
    #     st.markdown(f"**System:** {msg.content}")


# Load the drug recommendation chain and related system components.
@st.cache_resource
def load_system():
    # 1) Load documents from CSV using the OncologyDataLoader.
    documents = OncologyDataLoader(Config.CSV_PATH).load_data()

    # 2) Create embeddings.
    embeddings = OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        openai_api_key=Config.OPENAI_API_KEY
    )

    # 3) Build the FAISS vector store.
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 4) Initialize the LLM for drug recommendation.
    llm = ChatOpenAI(
        temperature=0,
        model_name=Config.LLM_MODEL,
        openai_api_key=Config.OPENAI_API_KEY
    )

    # 5) Create your custom drug recommendation chain.
    drug_chain = DrugRecommendationChain(retriever, llm)
    return drug_chain


# Text input for the user's response - use st.chat_input instead of text_input
user_input = st.chat_input("Your response:")

if user_input:
    # Append the user's response as a HumanMessage.
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Count the number of human responses (i.e. conversation turns).
    num_turns = sum(1 for m in st.session_state.messages if isinstance(m, HumanMessage))

    # If less than 5 turns, generate the next question.
    if num_turns < 5:
        # Use the conversation history as context.
        # chat_model.invoke() accepts a list of messages and returns an AIMessage.
        with st.spinner("Thinking..."):
            next_question_msg = chat_model.invoke(st.session_state.messages)
            response_content = next_question_msg.content
            st.session_state.messages.append(AIMessage(content=response_content))
            st.chat_message("assistant").write(response_content)
    else:
        # Final stage: Generate a final query from the conversation.
        with st.spinner("Analyzing patient data and generating drug recommendations..."):
            final_prompt = "Based on the following conversation, generate a final query summarizing the patient data for drug recommendation:\n"
            for msg in st.session_state.messages:
                if isinstance(msg, HumanMessage):
                    final_prompt += f"User: {msg.content}\n"
                elif isinstance(msg, AIMessage) and not isinstance(msg, SystemMessage):
                    final_prompt += f"Assistant: {msg.content}\n"

            try:
                # Load the drug recommendation chain (cached).
                drug_chain = load_system()
                # Generate the patient summary
                with st.status("Step 1: Summarizing patient data..."):
                    patient_summary = chat_model.invoke([SystemMessage(
                        content="Summarize the patient information from this conversation into a concise query for drug recommendation."),
                                                         HumanMessage(content=final_prompt)])

                # Use the summary to get drug recommendations
                with st.status("Step 2: Retrieving drug recommendations..."):
                    recommendation = drug_chain.invoke(patient_summary.content)

                # Display the recommendation
                st.session_state.messages.append(AIMessage(content=recommendation))
                st.chat_message("assistant").write(recommendation)
            except Exception as e:
                error_message = "I apologize, but I couldn't generate a drug recommendation. Please try refining your inputs."
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append(AIMessage(content=error_message))
                st.chat_message("assistant").write(error_message)

if st.sidebar.button("Start New Conversation"):
    st.session_state.messages = [initial_system_msg]
    # Add a welcome message
    welcome_message = AIMessage(
        content="Hello! I'm your oncology treatment assistant. I'll help you find the most appropriate FDA-approved treatment options based on your situation. Let's start with some questions about the patient. What is the patient's age and gender?")
    st.session_state.messages.append(welcome_message)
    st.rerun()