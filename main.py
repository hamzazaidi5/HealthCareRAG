import streamlit as st
from utils.config import Config
from chain.custom_chain import DrugRecommendationChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from utils.data_loader import OncologyDataLoader

# Initialize a ChatOpenAI instance for generating questions.
question_llm = ChatOpenAI(
    temperature=0,
    model_name=Config.LLM_MODEL,
    openai_api_key=Config.OPENAI_API_KEY
)


def generate_generic_question(stage, patient_data):
    """
    Generate a generic question based on the conversation stage and collected patient data.
    """
    if stage == 0:
        prompt = (
            "You are a friendly, professional medical chatbot. "
            "Greet the user and ask for their age, gender, and any relevant health history in a concise, generic manner."
        )
    elif stage == 1:
        prompt = (
            f"The user has provided basic demographics: {patient_data.get('demographics', '')}. "
            "Now, ask in a generic way: What type of cancer have you been diagnosed with?"
        )
    elif stage == 2:
        prompt = (
            f"The user mentioned their cancer type as: {patient_data.get('cancer_type', '')}. "
            "Ask generically: Do you know the stage of your cancer?"
        )
    elif stage == 3:
        prompt = (
            f"The user stated the cancer stage as: {patient_data.get('cancer_stage', '')}. "
            "Ask generically: Are there any known subtypes or genetic markers associated with your cancer (e.g., HER2, EGFR)?"
        )
    elif stage == 4:
        prompt = (
            f"The user provided subtype information: {patient_data.get('cancer_subtype', '')}. "
            "Ask generically: What treatments have you received so far?"
        )
    elif stage == 5:
        prompt = (
            "Now that all necessary patient data has been collected, "
            "generate a concise final query summarizing the patient data to identify the best FDA-approved drug(s) and available survival data."
        )
    elif stage == 6:
        prompt = (
            f"The user has a follow-up question: {patient_data.get('followup', '')}. "
            "Generate a clarifying generic follow-up response that addresses this query."
        )
    else:
        prompt = "Could you please provide additional details?"

    response = question_llm(prompt)
    return response.content.strip()


def initialize_system():
    # 1) Load your CSV as Documents (entire file or limited rows based on need)
    documents = OncologyDataLoader(Config.CSV_PATH).load_data()

    # 2) Create embeddings using your selected model and API key.
    embeddings = OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        openai_api_key=Config.OPENAI_API_KEY
    )

    # 3) Build the FAISS vector store for efficient document retrieval.
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 4) Initialize the LLM for recommendation generation.
    llm = ChatOpenAI(
        temperature=0,
        model_name=Config.LLM_MODEL,
        openai_api_key=Config.OPENAI_API_KEY
    )

    # 5) Create your custom drug recommendation chain.
    drug_chain = DrugRecommendationChain(retriever, llm)
    return drug_chain, llm


# Cache the system initialization so that the vector store and LLM load only once.
@st.cache_resource
def load_system():
    return initialize_system()


def main():
    st.title("Oncology Treatment Information Assistant")

    # Initialize conversation states.
    if "stage" not in st.session_state:
        st.session_state.stage = 0
    if "patient_data" not in st.session_state:
        st.session_state.patient_data = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Load the recommendation chain system and the llm instance.
    drug_chain, llm = load_system()

    # When starting a new conversation, automatically generate the initial generic question.
    if st.session_state.stage == 0 and not st.session_state.chat_history:
        initial_question = generate_generic_question(0, st.session_state.patient_data)
        st.session_state.chat_history.append({"role": "assistant", "content": initial_question})

    # Display the conversation history.
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Assistant:** {message['content']}")

    user_input = st.text_input("Your response:", key="user_message")
    if st.button("Send"):
        # Append user input.
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Process response based on conversation stage.
        if st.session_state.stage == 0:
            # Save demographics (age/gender/history).
            st.session_state.patient_data["demographics"] = user_input
            next_q = generate_generic_question(1, st.session_state.patient_data)
            st.session_state.chat_history.append({"role": "assistant", "content": next_q})
            st.session_state.stage = 1

        elif st.session_state.stage == 1:
            st.session_state.patient_data["cancer_type"] = user_input
            next_q = generate_generic_question(2, st.session_state.patient_data)
            st.session_state.chat_history.append({"role": "assistant", "content": next_q})
            st.session_state.stage = 2

        elif st.session_state.stage == 2:
            st.session_state.patient_data["cancer_stage"] = user_input
            next_q = generate_generic_question(3, st.session_state.patient_data)
            st.session_state.chat_history.append({"role": "assistant", "content": next_q})
            st.session_state.stage = 3

        elif st.session_state.stage == 3:
            st.session_state.patient_data["cancer_subtype"] = user_input
            next_q = generate_generic_question(4, st.session_state.patient_data)
            st.session_state.chat_history.append({"role": "assistant", "content": next_q})
            st.session_state.stage = 4

        elif st.session_state.stage == 4:
            st.session_state.patient_data["treatment_history"] = user_input

            # Generate final query using a generic prompt.
            final_query = generate_generic_question(5, st.session_state.patient_data)
            try:
                recommendation = drug_chain.invoke(final_query)
                st.session_state.chat_history.append({"role": "assistant", "content": recommendation})
            except Exception:
                error_msg = (
                    "I apologize, but I couldn't find specific information to make a recommendation. "
                    "Please try refining your inputs or ask about a different scenario."
                )
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            st.session_state.stage = 5

        elif st.session_state.stage == 5:
            # For follow-up queries, store the follow-up and generate a generic follow-up response.
            st.session_state.patient_data["followup"] = user_input
            followup_prompt = generate_generic_question(6, st.session_state.patient_data)
            try:
                followup_answer = drug_chain.invoke(followup_prompt)
                st.session_state.chat_history.append({"role": "assistant", "content": followup_answer})
            except Exception:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "I couldn't find additional information. Please clarify your question or ask about another treatment."
                })

        st.rerun()  # Update the UI.

    if st.button("Start New Conversation"):
        st.session_state.stage = 0
        st.session_state.patient_data = {}
        st.session_state.chat_history = []
        st.rerun()


if __name__ == "__main__":
    main()
