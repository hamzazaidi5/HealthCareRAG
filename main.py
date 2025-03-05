import streamlit as st
import random
from utils.config import Config
from chain.custom_chain import DrugRecommendationChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import re

from utils.data_loader import OncologyDataLoader


# Humanization Helper Functions
def get_thinking_phrases():
    return [
        "Let me think about that...",
        "Hmm, let me gather my thoughts...",
        "Interesting! Give me a moment to process...",
        "Let me put on my oncology detective hat...",
        "Analyzing the details carefully...",
        "Processing your information...",
        "Consulting my medical knowledge base...",
        "Just a sec while I review the details...",
    ]


def get_empathetic_intros():
    return [
        "I appreciate you sharing this important information. ",
        "Thank you for providing those details. ",
        "I'm carefully considering the information you've shared. ",
        "Your input is crucial for making the best recommendation. ",
    ]


def get_follow_up_questions():
    return [
        "To help me provide the most accurate recommendations, ",
        "To ensure we cover all the important details, ",
        "To get a complete picture of the patient's condition, ",
    ]


# Initialize ChatOpenAI instance used for generating questions.
chat_model = ChatOpenAI(
    temperature=0,
    model_name=Config.LLM_MODEL,
    openai_api_key=Config.OPENAI_API_KEY
)

# Define an initial system message that sets the conversation context with a more human touch
initial_system_msg = SystemMessage(
    content=(
        "You are a compassionate, professional oncology treatment assistant. "
        "Communicate in a warm, supportive manner while maintaining medical accuracy. "
        "Your goal is to gather key patient information through friendly, targeted questions. "
        "Use empathetic language and show genuine care in your interactions."
    )
)

# Initialize conversation messages in session state using LangChain message objects.
if "messages" not in st.session_state:
    st.session_state.messages = [initial_system_msg]
    # Add a more conversational welcome message
    welcome_message = AIMessage(
        content=(
            "Hi there! 👋 I'm your oncology treatment support assistant. 🩺 "
            "I'm here to help find the most appropriate treatment options with care and precision. "
            "Let's start by getting to know a bit about the patient. "
            "Could you tell me their age, gender, and type of cancer?"
        )
    )
    st.session_state.messages.append(welcome_message)

# Initialize turn counter if not exists
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0

# Track if we've asked all questions
if "questions_complete" not in st.session_state:
    st.session_state.questions_complete = False

# Track discovered cancer type
if "cancer_type" not in st.session_state:
    st.session_state.cancer_type = None

# Display the conversation history with more engaging presentation
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        # Add a little randomness to make it feel more natural
        st.chat_message("assistant").write(msg.content)


# Function to extract cancer type from conversation
def extract_cancer_type(messages):
    # First, check if we already stored the cancer type
    if st.session_state.cancer_type:
        return st.session_state.cancer_type

    # Create a prompt to extract cancer type
    extraction_prompt = "Based on the conversation, extract only the cancer type mentioned. Reply with just the cancer type name, nothing else. If no specific cancer is mentioned, reply with 'Unknown'."

    # Create a temporary list of relevant messages
    relevant_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
            relevant_messages.append(msg)

    # Ask the model to extract the cancer type
    try:
        result = chat_model.invoke([
            SystemMessage(content=extraction_prompt),
            *relevant_messages
        ])
        cancer_type = result.content.strip()

        # Store it for future use
        if cancer_type and cancer_type.lower() != "unknown":
            st.session_state.cancer_type = cancer_type

        return cancer_type
    except Exception as e:
        print(f"Error extracting cancer type: {str(e)}")
        return "Unknown"


# Load the drug recommendation chain and related system components
@st.cache_resource
def load_system():
    # 1) Load documents from CSV using the OncologyDataLoader
    documents = OncologyDataLoader(Config.CSV_PATH).load_data()

    # 2) Create embeddings
    embeddings = OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        openai_api_key=Config.OPENAI_API_KEY
    )

    # 3) Build the FAISS vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 4) Initialize the LLM for drug recommendation
    llm = ChatOpenAI(
        temperature=0,
        model_name=Config.LLM_MODEL,
        openai_api_key=Config.OPENAI_API_KEY
    )

    # 5) Create your custom drug recommendation chain
    drug_chain = DrugRecommendationChain(retriever, llm)
    return drug_chain


# Text input for the user's response
user_input = st.chat_input("Your response...")

if user_input:
    # Append the user's response as a HumanMessage
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Increment turn count
    st.session_state.turn_count += 1

    # Extract cancer type after each user input
    extract_cancer_type(st.session_state.messages)

    # Determine if we should generate recommendations
    last_ai_message = next((msg for msg in reversed(st.session_state.messages)
                            if isinstance(msg, AIMessage)), None)

    final_question_indicator = last_ai_message and any(phrase in last_ai_message.content.lower()
                                                       for phrase in
                                                       ["final question", "last question", "one more question"])

    enough_turns = st.session_state.turn_count >= 4

    if final_question_indicator or enough_turns or st.session_state.questions_complete:
        st.session_state.questions_complete = True

        # Add a human-like thinking message
        st.chat_message("assistant").write(random.choice(get_thinking_phrases()))

        # Final stage: Generate drug recommendations
        with st.spinner("Carefully analyzing patient information..."):
            # Extract the conversation for context
            conversation_context = ""
            for msg in st.session_state.messages:
                if isinstance(msg, HumanMessage):
                    conversation_context += f"User: {msg.content}\n"
                elif isinstance(msg, AIMessage) and not isinstance(msg, SystemMessage):
                    conversation_context += f"Assistant: {msg.content}\n"

            # Generate a comprehensive patient summary with emphasis on cancer type
            cancer_type = extract_cancer_type(st.session_state.messages)

            # Create a specialized prompt that emphasizes the exact cancer type
            summary_prompt = f"""
            Based on this conversation, create a detailed patient summary for drug recommendation.
            The patient has been diagnosed with {cancer_type}.
            Include all relevant details mentioned like age, gender, stage, prior treatments, etc.
            Format as a concise paragraph focusing on clinical details relevant for treatment decisions.

            Conversation:
            {conversation_context}
            """

            try:
                # Generate the patient summary
                with st.status("Preparing personalized recommendations..."):
                    st.write("1. Carefully reviewing patient information")
                    patient_summary = chat_model.invoke([
                        SystemMessage(content="You are a clinical oncologist creating a precise patient summary."),
                        HumanMessage(content=summary_prompt)
                    ])

                    # Explicitly add the cancer type to the summary for emphasis
                    enhanced_summary = f"Patient has {cancer_type}. " + patient_summary.content

                    st.write("2. Identifying potential treatment options")
                    # Load the drug recommendation chain (cached)
                    drug_chain = load_system()

                    st.write("3. Generating final personalized recommendations")
                    # Use the summary to get drug recommendations
                    recommendation = drug_chain.invoke(enhanced_summary)

                # Add an empathetic introduction to recommendations
                empathetic_intro = random.choice(get_empathetic_intros())
                final_recommendation = f"{empathetic_intro}\n\n{recommendation}"

                # Add a supportive closing note
                final_recommendation += (
                    "\n\n💕 Remember, every patient's journey is unique. "
                    "These recommendations are based on the latest clinical evidence, "
                    "but always consult with your healthcare team for personalized advice."
                )

                # Display and store the recommendation
                st.session_state.messages.append(AIMessage(content=final_recommendation))
                st.chat_message("assistant").write(final_recommendation)

                # Reset questions complete for future interactions
                st.session_state.questions_complete = True

            except Exception as e:
                error_message = (
                    f"I apologize, but I couldn't generate a drug recommendation for {cancer_type}. "
                    "This might be due to limited information or complexity of the case. "
                    "Would you like to provide more details to help me understand better?"
                )
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append(AIMessage(content=error_message))
                st.chat_message("assistant").write(error_message)

    else:
        # Continue asking questions with a more conversational approach
        with st.spinner("Thinking carefully..."):
            # Add a conversational thinking pause
            st.chat_message("assistant").write(random.choice(get_thinking_phrases()))

            # Replace the existing guidance_msg in the question generation section with this:
            guidance_msg = SystemMessage(
                content=f"""
                Interaction {st.session_state.turn_count} of 4.
                Current information: {extract_cancer_type(st.session_state.messages) if st.session_state.cancer_type else "Basic info"}

                Generate a BRIEF, direct question to gather ONE key piece of medical information.

                Constraints:
                - Maximum 15 words
                - Sound professional and caring
                - Focus on critical clinical details
                - Ensure the question is specific and actionable
                """
            )
            # Add the guidance message temporarily for this response
            temp_messages = st.session_state.messages + [guidance_msg]

            # Modify the question processing logic
            next_question_msg = chat_model.invoke(temp_messages)
            response_content = next_question_msg.content

            # Ensure the question is concise and ends with a question mark
            if len(response_content) > 100:
                response_content = response_content[:100]

            if not response_content.strip().endswith('?'):
                response_content += "?"

            # If no specific question is generated, use a fallback
            if len(response_content) < 10:
                fallback_questions = [
                    "What stage is the breast cancer?",
                    "Have any specific tests been done?",
                    "Are there any known symptoms?",
                    "What is the patient's treatment history?"
                ]
                response_content = fallback_questions[st.session_state.turn_count % len(fallback_questions)]

            # Remove the lengthy follow-up intros
            st.session_state.messages.append(AIMessage(content=response_content))
            st.chat_message("assistant").write(response_content)



# Sidebar button to start a new conversation with a friendly reset
if st.sidebar.button("Start New Conversation"):
    st.session_state.messages = [initial_system_msg]
    # Add a welcome message
    welcome_message = AIMessage(
        content=(
            "Hi there! 👋 Let's start fresh. "
            "I'm ready to help find the most appropriate treatment options. "
            "Could you tell me about the patient?"
        )
    )
    st.session_state.messages.append(welcome_message)
    st.session_state.turn_count = 0
    st.session_state.questions_complete = False
    st.session_state.cancer_type = None
    st.rerun()