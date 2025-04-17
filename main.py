import streamlit as st
import random
from utils.config import Config
from chain.custom_chain import DrugRecommendationChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import re
import pandas as pd

from utils.data_loader import OncologyDataLoader


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
        "For this case of advanced prostate cancer, ",
        "Given the clinical details provided, ",
        "Considering the stage of disease, ",
        "After reviewing all the information, ",
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


# Initialize ChatOpenAI instance used for generating questions.
chat_model = ChatOpenAI(
    temperature=0.4,  # Increased for more natural variation
    model_name=Config.LLM_MODEL,
    openai_api_key=Config.OPENAI_API_KEY
)

# Define an initial system message that sets the conversation context with a more human touch
initial_system_msg = SystemMessage(
    content=(
        "You are an experienced oncologist with a natural conversational style conducting a patient consultation. "
        "Your goal is to gather a complete clinical picture through thoughtful, connected questions. "
        "Vary your language patterns and avoid repetitive phrases like 'thank you for sharing'. "
        "Use medical terminology appropriately while remaining clear and compassionate. "
        "Approach the conversation as a flowing clinical dialogue rather than a structured interview. "
        "Reference previous information when asking follow-up questions to create continuity. "
        "Only acknowledge information when it feels natural, not after every response. "
        "Do not refer to yourself by name in your responses."
    )
)

# Initialize conversation messages in session state using LangChain message objects.
if "messages" not in st.session_state:
    st.session_state.messages = [initial_system_msg]
    # Add a more natural doctor welcome message
    welcome_message = AIMessage(
        content=(
            "Hello! I'm here to help you find both FDA-approved treatments and clinical trials "
            "that might be suitable for you. To start, could you tell me about your medical "
            "condition and where you're located?"
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

# Track patient information for contextual references
if "patient_info" not in st.session_state:
    st.session_state.patient_info = {
        "age": None,
        "gender": None,
        "cancer_type": None,
        "stage": None,
        "prior_treatments": [],
        "biomarkers": [],
        "comorbidities": []
    }

# Track conversation flow to avoid repetitive patterns
if "last_acknowledgment" not in st.session_state:
    st.session_state.last_acknowledgment = ""

# Display the conversation history with more engaging presentation
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)


def generate_extraction_prompt():
    # Start with the basic structure of the extraction prompt
    extraction_prompt = """
    Based on the conversation, extract the following patient information. 
    For each field, provide ONLY the value and nothing else. If a value is unknown, reply with 'Unknown'.

    Format your response exactly like this:

    Cancer Type: [cancer type]
    Age: [age]
    Gender: [gender]
    Stage: [stage]
    Prior Treatments: [comma-separated list]
    Biomarkers: [comma-separated list]
    Comorbidities: [comma-separated list]
    """

    # Add specific questions based on cancer type (dynamic)
    if st.session_state.cancer_type:
        if "breast" in st.session_state.cancer_type.lower():
            extraction_prompt += "\nFor breast cancer, is it HER2+? Please mention any other biomarker statuses (e.g., ER+, PR+)."
        elif "lung" in st.session_state.cancer_type.lower():
            extraction_prompt += "\nFor lung cancer, has the tumor been tested for EGFR, ALK, or PD-L1 mutations?"
        # Add other conditions here based on cancer type

    return extraction_prompt


# Update the extraction process with dynamic prompt generation
extraction_prompt = generate_extraction_prompt()


# Enhanced function to extract patient information from conversation
def extract_patient_info(messages):
    # First, check if we already identified the cancer type
    if st.session_state.cancer_type:
        st.session_state.patient_info["cancer_type"] = st.session_state.cancer_type
        return st.session_state.patient_info

    # Dynamically generate extraction prompt
    extraction_prompt = generate_extraction_prompt()

    # Create a temporary list of relevant messages
    relevant_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
            relevant_messages.append(msg)

    # Ask the model to extract the patient information
    try:
        result = chat_model.invoke([
            SystemMessage(content=extraction_prompt),
            *relevant_messages
        ])

        # Parse the structured response
        info_text = result.content.strip()

        # Extract cancer type
        cancer_match = re.search(r"Cancer Type: (.+)$", info_text, re.MULTILINE)
        if cancer_match and cancer_match.group(1).lower() != "unknown":
            cancer_type = cancer_match.group(1).strip()
            st.session_state.cancer_type = cancer_type
            st.session_state.patient_info["cancer_type"] = cancer_type

        # Extract age
        age_match = re.search(r"Age: (.+)$", info_text, re.MULTILINE)
        if age_match and age_match.group(1).lower() != "unknown":
            st.session_state.patient_info["age"] = age_match.group(1).strip()

        # Extract gender
        gender_match = re.search(r"Gender: (.+)$", info_text, re.MULTILINE)
        if gender_match and gender_match.group(1).lower() != "unknown":
            st.session_state.patient_info["gender"] = gender_match.group(1).strip()

        # Extract stage
        stage_match = re.search(r"Stage: (.+)$", info_text, re.MULTILINE)
        if stage_match and stage_match.group(1).lower() != "unknown":
            st.session_state.patient_info["stage"] = stage_match.group(1).strip()

        # Extract prior treatments
        treatments_match = re.search(r"Prior Treatments: (.+)$", info_text, re.MULTILINE)
        if treatments_match and treatments_match.group(1).lower() != "unknown":
            treatments = [t.strip() for t in treatments_match.group(1).split(",")]
            st.session_state.patient_info["prior_treatments"] = treatments

        # Extract biomarkers
        biomarkers_match = re.search(r"Biomarkers: (.+)$", info_text, re.MULTILINE)
        if biomarkers_match and biomarkers_match.group(1).lower() != "unknown":
            biomarkers = [b.strip() for b in biomarkers_match.group(1).split(",")]
            st.session_state.patient_info["biomarkers"] = biomarkers

        # Extract comorbidities
        comorbidities_match = re.search(r"Comorbidities: (.+)$", info_text, re.MULTILINE)
        if comorbidities_match and comorbidities_match.group(1).lower() != "unknown":
            comorbidities = [c.strip() for c in comorbidities_match.group(1).split(",")]
            st.session_state.patient_info["comorbidities"] = comorbidities

        return st.session_state.patient_info

    except Exception as e:
        print(f"Error extracting patient information: {str(e)}")
        return st.session_state.patient_info




def get_matching_trials(cancer_type: str, path="data/Active_Recruiting_Trials.csv") -> pd.DataFrame:
    try:
        # Read the CSV file
        df = pd.read_csv(path)

        # Clean up column names by stripping any leading/trailing whitespace
        df.columns = df.columns.str.strip()

        # Print the column names to debug
        print("Columns in the DataFrame:", df.columns)

        # Ensure 'Conditions' column exists
        if 'Conditions' not in df.columns:
            print("Error: 'Conditions' column not found.")
            return pd.DataFrame()  # Return empty DataFrame if 'Conditions' column is missing

        # Normalize user input (make it lowercase)
        cancer_type_normalized = cancer_type.lower().strip()

        # Filter rows where 'Conditions' contains exactly the cancer type and no other types (no '|')
        # Match the cancer type exactly with no other conditions in the string (no '|')
        filtered = df[
            df['Conditions'].str.lower().str.contains(cancer_type_normalized, na=False) &
            ~df['Conditions'].str.contains(r'\|', na=False)  # Exclude rows with '|' (multiple conditions)
        ]

        print(filtered, "filtered")
        return filtered

    except Exception as e:
        print(e, "not found")
        return pd.DataFrame()


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

    # Extract patient information after each user input
    extract_patient_info(st.session_state.messages)

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
        with st.spinner("Analyzing clinical information..."):
            # Extract the conversation for context
            conversation_context = ""
            for msg in st.session_state.messages:
                if isinstance(msg, HumanMessage):
                    conversation_context += f"User: {msg.content}\n"
                elif isinstance(msg, AIMessage) and not isinstance(msg, SystemMessage):
                    conversation_context += f"Doctor: {msg.content}\n"

            # Generate a comprehensive patient summary with emphasis on cancer type
            patient_info = extract_patient_info(st.session_state.messages)
            cancer_type = patient_info["cancer_type"] or "Unknown cancer type"

            # Create a specialized prompt that uses medical terminology
            summary_prompt = f"""
            Based on this consultation, create a detailed clinical summary for treatment planning.
            The patient has been diagnosed with {cancer_type}.
            Include all relevant clinical details mentioned such as age, gender, disease stage, prior treatments, biomarkers, comorbidities, etc.
            Format as a concise medical assessment focusing on details relevant for treatment decision-making.

            Consultation transcript:
            {conversation_context}
            """

            try:
                # Generate the patient summary
                with st.status("Preparing clinical recommendations..."):
                    st.write("1. Reviewing clinical data")
                    patient_summary = chat_model.invoke([
                        SystemMessage(content="You are an oncologist creating a precise clinical assessment."),
                        HumanMessage(content=summary_prompt)
                    ])

                    # Explicitly add the cancer type to the summary for emphasis
                    enhanced_summary = f"Patient has {cancer_type}. " + patient_summary.content

                    st.write("2. Evaluating evidence-based treatment options")
                    # Load the drug recommendation chain (cached)
                    drug_chain = load_system()

                    st.write("3. Generating personalized treatment plan")
                    # Use the summary to get drug recommendations
                    recommendation = drug_chain.invoke(enhanced_summary)

                # Add a clinical introduction to recommendations
                empathetic_intro = random.choice(get_empathetic_intros())
                final_recommendation = f"{empathetic_intro}\n\n{recommendation}"

                trials_section = ""  # Initialize the trials section string
                trial_count = 1  # Initialize a counter for numbering the trials
                matching_trials = get_matching_trials(cancer_type)
                # Loop through the first 3 matching trials and add them to trials_section
                for _, row in matching_trials.head(3).iterrows():  # Only process the first 3 trials
                    # Format the trial information with one newline after each section (no double newlines)
                    trials_section += (
                         f"\n\n**Trial {trial_count}:**\n\n"  # Display the trial number (1, 2, 3, ...)
                        f"**Cancer Type**: {row.get('Conditions', 'N/A')}\n\n"  # Cancer Type (one line break)
                        f"**Location**: {row.get('Locations', 'No description available.')}\n\n"  # Location
                        f"**Age**: {row.get('Age', 'N/A')}\n\n"  # Age
                        f"**Sex**: {row.get('Sex', 'N/A')}\n\n"  # Sex
                        f"**Start Date**: {row.get('Start Date', 'N/A')}\n\n"  # Start Date
                        f"**Primary Completion Date**: {row.get('Primary Completion Date', 'N/A')}\n\n"  # Primary Completion Date
                        f"**Completion Date**: {row.get('Completion Date', 'N/A')}\n\n"  # Completion Date
                        "\n---\n"  # Separator between trials
                    )
                    trial_count += 1  # Increment the counter for the next trial

                # Append the trials section to the final recommendation
                final_recommendation += trials_section

                # Add a supportive closing note with a doctor's perspective
                final_recommendation += (
                    "\n\nIt's important to consider these recommendations in the context "
                    "of the patient's overall health status and preferences. While these options are supported by "
                    "clinical evidence, the final treatment decision should be made after discussion of potential "
                    "benefits and risks with the patient."
                    "\n\nWould you like me to elaborate on any particular aspect of the treatment plan?"
                )

                # Display and store the recommendation
                st.session_state.messages.append(AIMessage(content=final_recommendation))
                st.chat_message("assistant").write(final_recommendation)

                # Reset questions complete for future interactions
                st.session_state.questions_complete = True

            except Exception as e:
                error_message = (
                    f"I'm unable to formulate a complete treatment recommendation for {cancer_type} at this time. "
                    "This may be due to insufficient clinical information or the complexity of the case. "
                    "Could you provide additional details about the patient's disease characteristics or relevant biomarkers?"
                )
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append(AIMessage(content=error_message))
                st.chat_message("assistant").write(error_message)

    else:
        # Continue asking questions with a more natural doctor approach
        with st.spinner("Reviewing information..."):
            # Add a clinical thinking pause occasionally (20% of the time)
            if random.random() < 0.2:
                st.chat_message("assistant").write(random.choice(get_thinking_phrases()))

            # Get current patient info for context
            patient_info = extract_patient_info(st.session_state.messages)

            # Create a context-aware guidance message for question generation
            patient_context = ""
            if patient_info["age"]:
                patient_context += f"Patient age: {patient_info['age']}. "
            if patient_info["gender"]:
                patient_context += f"Patient gender: {patient_info['gender']}. "
            if patient_info["cancer_type"]:
                patient_context += f"Cancer type: {patient_info['cancer_type']}. "
            if patient_info["stage"]:
                patient_context += f"Disease stage: {patient_info['stage']}. "

            # Add the prior treatments
            if patient_info["prior_treatments"] and len(patient_info["prior_treatments"]) > 0:
                treatments = ", ".join(patient_info["prior_treatments"])
                if treatments.lower() != "unknown":
                    patient_context += f"Prior treatments: {treatments}. "

            guidance_msg = SystemMessage(
                content=f"""
            You are simulating a compassionate oncology specialist conducting a conversational consultation with a patient to gather key clinical information. Your tone should be supportive, knowledgeable, and patient-centered.

            Consultation step {st.session_state.turn_count} of 4.

            Known patient information: {patient_context if patient_context else "Initial consultation"}

            🎯 Objective:
            Generate a thoughtful, natural follow-up message or question to guide the clinical intake conversation. Use the patient's previous responses as context.

            🧠 Your message must:
            1. Reference previously shared information when helpful
            2. Ask **multiple clinically relevant follow-up questions in one message**
            3. Prioritize gathering one or two **critical pieces of missing information** (see list below)
            4. Sound like a human physician, with natural flow, tone, and empathy
            5. Use professional but conversational medical language
            6. Vary structure and sentence starters to avoid robotic repetition

            ❌ Do NOT:
            - Use robotic phrasing like “Could you share…” or “What is…”
            - Say “Thank you for sharing” or “I appreciate your input”
            - Ask a single short question

            ✅ Do:
            - Use phrases like:
              - “Just to get a clearer picture…”
              - “Based on what you’ve shared so far…”
              - “Before we move on, I’d like to understand...”
              - “It might help to know…”

            📌 Clinical information to collect (prioritize only 1-2 per message):
            - Type of cancer and date of diagnosis
            - Cancer stage and spread (metastasis, lymph node involvement)
            - Biomarker status (e.g., PSA, HER2, EGFR)
            - Prior treatments and patient response
            - Comorbidities or performance status
            - Symptoms or lab results impacting treatment

            💬 Example output:
            “Since we’re talking about stage III prostate cancer, it would be helpful to know if you’ve had any biomarker testing done — things like PSA levels or genetic mutations like BRCA. Also, have you received any treatments so far, like hormone therapy or radiation? Knowing how you responded can guide us toward the most effective options.”

            Keep your message focused, warm, and inquisitive — you’re building rapport while gathering clinical info.
            """
            )

            # Track last interaction to avoid repetition
            last_human_msg = next((msg for msg in reversed(st.session_state.messages)
                                   if isinstance(msg, HumanMessage)), None)

            # Add the guidance message temporarily for this response
            temp_messages = st.session_state.messages + [guidance_msg]

            # Generate the next question with context
            next_question_msg = chat_model.invoke(temp_messages)
            response_content = next_question_msg.content

            # If no specific question is generated, use a contextual fallback
            if len(response_content.strip()) < 10:
                # Use patient context to create a relevant fallback
                if not patient_info["cancer_type"]:
                    response_content = "Could you specify the exact type and location of the cancer?"
                elif not patient_info["stage"]:
                    starter = random.choice(get_question_starters())
                    response_content = f"{starter}the stage of the {patient_info['cancer_type']}?"
                elif not patient_info["prior_treatments"] or len(patient_info["prior_treatments"]) == 0:
                    starter = random.choice(get_question_starters())
                    response_content = f"{starter}any previous treatments for the {patient_info['cancer_type']}?"
                else:
                    fallback_questions = [
                        f"Are there any biomarker test results for this {patient_info['cancer_type']}?",
                        "How would you describe the patient's current functional status?",
                        "Any other health conditions we should factor into the treatment plan?",
                        "What's most important to the patient regarding treatment goals?"
                    ]
                    response_content = fallback_questions[st.session_state.turn_count % len(fallback_questions)]

            # Check for repetitive thank you patterns and replace if found
            thank_you_patterns = [
                r"thank you for sharing",
                r"thank you for providing",
                r"thanks for sharing",
                r"I appreciate you sharing"
            ]

            for pattern in thank_you_patterns:
                if re.search(pattern, response_content, re.IGNORECASE):
                    # Replace with a more natural acknowledgment or just remove
                    random_ack = random.choice(get_acknowledgment_phrases())
                    response_content = re.sub(pattern, random_ack, response_content, flags=re.IGNORECASE)

            # Check for self-reference by name and remove
            response_content = re.sub(r"(?i)Dr\.\s*Carter", "", response_content)
            response_content = re.sub(r"(?i)doctor\s*Carter", "", response_content)

            # Store and display the response
            st.session_state.messages.append(AIMessage(content=response_content))
            st.chat_message("assistant").write(response_content)

# Sidebar with a more clinical framing
st.sidebar.title("Oncology Consultation")
st.sidebar.markdown("---")

if st.sidebar.button("Start New Consultation"):
    st.session_state.messages = [initial_system_msg]
    # Add a welcome message
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
        "gender": None,
        "cancer_type": None,
        "stage": None,
        "prior_treatments": [],
        "biomarkers": [],
        "comorbidities": []
    }
    st.rerun()