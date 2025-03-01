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

# initial_system_msg = SystemMessage(
#     content=(
#         "You are a friendly, professional oncology treatment assistant. "
#         "Your task is to ask the user a series of up to 5 specific questions to collect key patient data. "
#         "Always start with age, gender, and cancer type. Then ask about stage, histology/subtype, biomarkers, "
#         "and treatment history. Keep your questions focused and specific. "
#         "After collecting this information, automatically proceed to generate treatment recommendations "
#         "without asking the user to wait or confirm."
#     )
# )

# Define an initial system message that sets the conversation context.
initial_system_msg = SystemMessage(
    content=(
        "You are a friendly, professional oncology treatment assistant. "
        "Your task is to ask the user a series of up to 5 generic questions to collect key patient data "
        "(e.g., demographics, cancer type, stage, subtype, treatment history). "
        "After collecting this information, automatically proceed to generate treatment recommendations "
        "without asking the user to wait or confirm."
  )
)

# Initialize conversation messages in session state using LangChain message objects.
if "messages" not in st.session_state:
    st.session_state.messages = [initial_system_msg]
    # Add a welcome message
    welcome_message = AIMessage(
        content="Hello! I'm your oncology treatment assistant. I'll help you find the most appropriate FDA-approved treatment options. To get started, please tell me the patient's age and gender, and what type of cancer they have.")
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

# Display the conversation history.
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    # System messages are hidden


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
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Increased from 3 to 5 for better context

    # 4) Initialize the LLM for drug recommendation.
    llm = ChatOpenAI(
        temperature=0,
        model_name=Config.LLM_MODEL,
        openai_api_key=Config.OPENAI_API_KEY
    )

    # 5) Create your custom drug recommendation chain.
    drug_chain = DrugRecommendationChain(retriever, llm)
    return drug_chain


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


# Text input for the user's response - use st.chat_input instead of text_input
user_input = st.chat_input("Your response:")

if user_input:
    # Append the user's response as a HumanMessage.
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Increment turn count
    st.session_state.turn_count += 1

    # Extract cancer type after each user input
    extract_cancer_type(st.session_state.messages)

    # If we've had 4 or more turns, or if the last question contained a "final question" marker
    last_ai_message = next((msg for msg in reversed(st.session_state.messages)
                            if isinstance(msg, AIMessage)), None)

    final_question_indicator = last_ai_message and any(phrase in last_ai_message.content.lower()
                                                       for phrase in
                                                       ["final question", "last question", "one more question"])

    enough_turns = st.session_state.turn_count >= 4  # Reduced from 5 to 4

    if final_question_indicator or enough_turns or st.session_state.questions_complete:
        st.session_state.questions_complete = True
        # Final stage: Generate drug recommendations
        with st.spinner("Analyzing patient data and generating drug recommendations..."):
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
                with st.status("Generating treatment recommendations..."):
                    st.write("1. Summarizing patient information")
                    patient_summary = chat_model.invoke([
                        SystemMessage(content="You are a clinical oncologist creating a precise patient summary."),
                        HumanMessage(content=summary_prompt)
                    ])

                    # Explicitly add the cancer type to the summary for emphasis
                    enhanced_summary = f"Patient has {cancer_type}. " + patient_summary.content

                    st.write("2. Retrieving FDA-approved drugs for this cancer type")
                    # Load the drug recommendation chain (cached)
                    drug_chain = load_system()

                    st.write("3. Generating final recommendation")
                    # Use the summary to get drug recommendations
                    recommendation = drug_chain.invoke(enhanced_summary)

                # Display the recommendation
                st.session_state.messages.append(AIMessage(content=recommendation))
                st.chat_message("assistant").write(recommendation)

                # Reset questions complete for future interactions
                st.session_state.questions_complete = True

            except Exception as e:
                error_message = f"I apologize, but I couldn't generate a drug recommendation for {cancer_type}. Please try refining your inputs."
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append(AIMessage(content=error_message))
                st.chat_message("assistant").write(error_message)

    else:
        # Continue asking questions
        with st.spinner("Thinking..."):
            # Prepare a system message that guides the model to ask relevant questions
            # and track progress toward getting all needed information
            guidance_msg = SystemMessage(
                content=f"""
                You are collecting patient information for cancer treatment recommendations.
                This is interaction {st.session_state.turn_count} of 4.

                Information already collected: {extract_cancer_type(st.session_state.messages) if st.session_state.cancer_type else "No cancer type identified yet"}

                Based on what has been shared so far, ask one specific question to gather critical missing information.
                Focus on: cancer type (if not yet provided), stage, biomarkers, histology/subtype, metastasis, and treatment history.

                If you've already collected substantial information (4+ data points), indicate this is your "final question" 
                before generating recommendations.

                Keep your question concise and focused - no more than 1-2 sentences.
                Do NOT ask the user to wait or say you'll generate recommendations soon - just ask your question.
                """
            )

            # Add the guidance message temporarily for this response
            temp_messages = st.session_state.messages + [guidance_msg]

            # Generate the next question
            next_question_msg = chat_model.invoke(temp_messages)
            response_content = next_question_msg.content

            # Check if this might be a closing message rather than a question
            if not any(q in response_content.lower() for q in ["?", "what", "how", "could you", "can you"]):
                # If it's not a question, force it to be one
                response_content = f"Thank you for that information. Could you please tell me about the patient's {['cancer stage', 'treatment history', 'biomarker status', 'symptoms'][st.session_state.turn_count % 4]}?"

            # Store and display
            st.session_state.messages.append(AIMessage(content=response_content))
            st.chat_message("assistant").write(response_content)

if st.sidebar.button("Start New Conversation"):
    st.session_state.messages = [initial_system_msg]
    # Add a welcome message
    welcome_message = AIMessage(
        content="Hello! I'm your oncology treatment assistant. I'll help you find the most appropriate FDA-approved treatment options. To get started, please tell me the patient's age and gender, and what type of cancer they have.")
    st.session_state.messages.append(welcome_message)
    st.session_state.turn_count = 0
    st.session_state.questions_complete = False
    st.session_state.cancer_type = None
    st.rerun()