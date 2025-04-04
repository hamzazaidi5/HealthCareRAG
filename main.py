import streamlit as st
import random
from utils.config import Config
from chain.custom_chain import ComprehensiveRecommendationChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import re

from utils.data_loader import DataLoader
from langchain.schema import Document


# Humanization Helper Functions
def get_thinking_phrases():
    return [
        "Considering the clinical picture...",
        "Evaluating the options...",
        "Analyzing this case...",
        "Reviewing the relevant guidelines...",
        "Thinking through the appropriate approach...",
        "Looking at the full clinical context...",
        "Weighing the treatment options...",
        "Taking a moment to assess...",
    ]


def get_empathetic_intros():
    return [
        "Based on what you've told me, ",
        "Given the clinical details provided, ",
        "Considering your situation, ",
        "After reviewing your information, ",
        "Taking into account your case, ",
    ]


def get_acknowledgment_phrases():
    return [
        "",  # Empty string for cases when no acknowledgment is needed
        "I see. ",
        "Understood. ",
        "Got it. ",
        "Alright. ",
        "Noted. ",
    ]


def get_question_starters():
    return [
        "What about ",
        "Can you tell me about ",
        "I'd like to know ",
        "How about ",
        "Could you share ",
        "Is there any information on ",
        "Do you know ",
    ]


def get_follow_up_question_starters():
    return [
        "What's ",
        "Could you clarify ",
        "How would you describe ",
        "Is there ",
        "Have there been ",
    ]


# Initialize ChatOpenAI instance
chat_model = ChatOpenAI(
    temperature=0.4,
    model_name=Config.LLM_MODEL,
    openai_api_key=Config.OPENAI_API_KEY
)

# Define initial system message
initial_system_msg = SystemMessage(
    content=(
        "You are an experienced oncology consultant helping patients find appropriate treatments and clinical trials. "
        "Your goal is to gather relevant information to match patients with suitable treatments and trials. "
        "Focus on understanding the patient's condition, location preferences, and specific needs. "
        "Use medical terminology appropriately while remaining clear and compassionate. "
        "Approach the conversation as a helpful guide rather than a structured interview. "
        "Reference previous information when asking follow-up questions to create continuity."
    )
)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = [initial_system_msg]
    welcome_message = AIMessage(
        content=(
            "Hello! I'm here to help you find both FDA-approved treatments and clinical trials "
            "that might be suitable for you. To start, could you tell me about your medical "
            "condition and where you're located?"
        )
    )
    st.session_state.messages.append(welcome_message)

if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0

if "questions_complete" not in st.session_state:
    st.session_state.questions_complete = False

if "cancer_type" not in st.session_state:
    st.session_state.cancer_type = None

if "patient_info" not in st.session_state:
    st.session_state.patient_info = {
        "age": None,
        "sex": None,
        "cancer_type": None,
        "stage": None,
        "prior_treatments": [],
        "biomarkers": [],
        "comorbidities": [],
        "location": None
    }

if "last_acknowledgment" not in st.session_state:
    st.session_state.last_acknowledgment = ""

# Display conversation history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)


def truncate_conversation_history(messages, max_messages=8):
    if len(messages) <= max_messages + 1:
        return messages
    return [messages[0]] + messages[-(max_messages):]


def extract_patient_info(messages):
    if st.session_state.cancer_type:
        st.session_state.patient_info["cancer_type"] = st.session_state.cancer_type
        return st.session_state.patient_info

    recent_messages = messages[-5:] if len(messages) > 5 else messages

    extraction_prompt = """
    Extract only the following information from the conversation. Use 'Unknown' if not found:
    Cancer Type:
    Stage:
    Location:
    Prior Treatments:
    Treatment Response:
    Current Status:
    Biomarkers:
    """

    try:
        result = chat_model.invoke([
            SystemMessage(content=extraction_prompt),
            *recent_messages
        ])

        info_text = result.content.strip()

        # Extract information using regex patterns
        patterns = {
            "cancer_type": r"Cancer Type: (.+)$",
            "stage": r"Stage: (.+)$",
            "location": r"Location: (.+)$",
            "prior_treatments": r"Prior Treatments: (.+)$",
            "biomarkers": r"Biomarkers: (.+)$"
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, info_text, re.MULTILINE)
            if match and match.group(1).lower() != "unknown":
                value = match.group(1).strip()
                if key in ["prior_treatments", "biomarkers"]:
                    st.session_state.patient_info[key] = [v.strip() for v in value.split(",")]
                else:
                    st.session_state.patient_info[key] = value
                    if key == "cancer_type":
                        st.session_state.cancer_type = value

        return st.session_state.patient_info

    except Exception as e:
        print(f"Error extracting patient information: {str(e)}")
        return st.session_state.patient_info


@st.cache_resource
def load_system():
    try:
        # Initialize data loader with both datasets
        loader = DataLoader(
            trials_path="data/Active Recruiting Trials.csv",
            oncology_path="data/oncology_survival_summaries.csv"
        )

        # Create embeddings
        embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY
        )

        # Extract patient info for filtering
        patient_info = st.session_state.patient_info

        # Load combined data with filters
        all_docs = loader.load_combined_data(
            cancer_type=patient_info.get('cancer_type'),
            location_filter=patient_info.get('location')
        )

        if not all_docs:
            print("Warning: No documents loaded. Check if data files exist and contain valid data.")
            all_docs = [Document(
                page_content="No matching data found. Please provide more specific information.",
                metadata={'source': 'placeholder'}
            )]

        # Build the FAISS vector store
        vector_store = FAISS.from_documents(all_docs, embeddings)
        retriever = vector_store.as_retriever(
            search_kwargs={"k": min(4, len(all_docs))},  # Ensure k doesn't exceed document count
            search_type="mmr"
        )

        # Initialize the LLM
        llm = ChatOpenAI(
            temperature=0,
            model_name=Config.LLM_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
            max_tokens=1000
        )

        # Create the comprehensive recommendation chain
        recommendation_chain = ComprehensiveRecommendationChain(retriever, llm)
        return recommendation_chain
    except Exception as e:
        print(f"Error in load_system: {str(e)}")
        return None


# User input handling
user_input = st.chat_input("Your response...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.turn_count += 1

    # Extract patient information
    extract_patient_info(st.session_state.messages)

    # Check if we should generate recommendations
    last_ai_message = next((msg for msg in reversed(st.session_state.messages)
                            if isinstance(msg, AIMessage)), None)

    final_question_indicator = last_ai_message and any(phrase in last_ai_message.content.lower()
                                                       for phrase in
                                                       ["final question", "last question", "one more question"])

    enough_turns = st.session_state.turn_count >= 4

    if final_question_indicator or enough_turns or st.session_state.questions_complete:
        st.session_state.questions_complete = True

        st.chat_message("assistant").write(random.choice(get_thinking_phrases()))

        with st.spinner("Analyzing treatment options and clinical trials..."):
            try:
                recommendation_chain = load_system()

                if recommendation_chain is None:
                    raise Exception("Failed to initialize recommendation system")

                patient_info = extract_patient_info(st.session_state.messages)

                query = (
                    f"Patient with {patient_info['cancer_type']} "
                    f"stage {patient_info['stage']} "
                    f"located in {patient_info['location']}. "
                    f"Previous treatments: {', '.join(patient_info['prior_treatments'])}. "
                    f"Biomarkers: {', '.join(patient_info['biomarkers'])}. "
                    "Need both FDA-approved treatment options and clinical trials."
                )

                recommendations = recommendation_chain.invoke(query)

                st.session_state.messages.append(AIMessage(content=recommendations))
                st.chat_message("assistant").write(recommendations)

            except Exception as e:
                error_message = (
                    "I apologize, but I'm having trouble processing your information. "
                    "This could be due to:\n"
                    "1. Missing critical information about your condition\n"
                    "2. Technical limitations in processing the data\n"
                    "3. Connectivity issues with our knowledge base\n\n"
                    "Please try starting a new consultation or provide more specific details about your condition."
                )
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append(AIMessage(content=error_message))
                st.chat_message("assistant").write(error_message)

    else:
        with st.spinner("Reviewing information..."):
            if random.random() < 0.2:
                st.chat_message("assistant").write(random.choice(get_thinking_phrases()))

            patient_info = extract_patient_info(st.session_state.messages)

            # Create context for next question
            patient_context = " ".join(
                f"{key}: {value}. " for key, value in patient_info.items()
                if value and value != [] and value != "Unknown"
            )

            guidance_msg = SystemMessage(
                content=f"""
                Consultation step {st.session_state.turn_count} of 4.
                Current information: {patient_context}

                Generate the next logical question focusing on missing critical information:
                - Disease stage if not known
                - Biomarker status
                - Prior treatments and response
                - Current symptoms and status

                Keep the tone professional but conversational.
                Avoid repetitive acknowledgments.
                Reference previous information when appropriate.
                """
            )

            temp_messages = truncate_conversation_history(st.session_state.messages + [guidance_msg])

            next_question = chat_model.invoke(temp_messages)
            response_content = next_question.content

            # Store and display the response
            st.session_state.messages.append(AIMessage(content=response_content))
            st.chat_message("assistant").write(response_content)

# Sidebar
st.sidebar.title("Oncology Consultation")
st.sidebar.markdown("---")

if st.sidebar.button("Start New Consultation"):
    st.session_state.messages = [initial_system_msg]
    welcome_message = AIMessage(
        content=(
            "Hello! I'm here to help you find both FDA-approved treatments and clinical trials "
            "that might be suitable for you. To start, could you tell me about your medical "
            "condition and where you're located?"
        )
    )
    st.session_state.messages.append(welcome_message)
    st.session_state.turn_count = 0
    st.session_state.questions_complete = False
    st.session_state.cancer_type = None
    st.session_state.last_acknowledgment = ""
    st.session_state.patient_info = {
        "age": None,
        "sex": None,
        "cancer_type": None,
        "stage": None,
        "prior_treatments": [],
        "biomarkers": [],
        "comorbidities": [],
        "location": None
    }
    st.rerun()