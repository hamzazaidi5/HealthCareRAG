import streamlit as st
import random
from utils.config import Config
from chain.custom_chain import DrugRecommendationChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import re
import pandas as pd
import os

from utils.data_loader import OncologyDataLoader

st.markdown("""
    <style>
    .chat-left {
        text-align: left;
        background-color: transparent;
        padding: 12px;
        border-radius: 12px;
        margin: 10px 0;
        color: white;
        max-width: 70%;
        border: 1px solid transparent;
    }

    .chat-right {
        text-align: right;
        background-color: #1e293b;
        padding: 12px;
        border-radius: 12px;
        margin: 10px 0;
        color: white;
        max-width: 70%;
        align-self: flex-end;
        margin-left: auto;
        border: 1px solid #334155;
    }

    .chat-wrapper {
        display: flex;
        flex-direction: column;
    }
    </style>
""", unsafe_allow_html=True)

def chat_bubble(text, sender="assistant"):
    css_class = "chat-left" if sender == "assistant" else "chat-right"
    st.markdown(f"""
        <div class="chat-wrapper">
            <div class="{css_class}">{text}</div>
        </div>
    """, unsafe_allow_html=True)

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
            "Hi there, I’m designed to use AI to help you make better treatment decisions."
        "Kindly share you age, gender and diagnosis? (If you prefer not to share details that’s ok for now)"
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
        chat_bubble(msg.content, sender="user")
    elif isinstance(msg, AIMessage):
        chat_bubble(msg.content, sender="assistant")


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


def get_matching_trials(cancer_type: str, path="data/Active_Recruiting_Trials.xls") -> pd.DataFrame:
    try:
        print("Finding trials for:", cancer_type)
        file_ext = os.path.splitext(path)[1].lower()

        if file_ext in ['.xls', '.xlsx']:
            df = pd.read_excel(path, engine='xlrd' if file_ext == '.xls' else 'openpyxl')
        elif file_ext == '.csv':
            df = pd.read_csv(path)
        else:
            print(f"Unsupported file format: {file_ext}")
            return pd.DataFrame()

        df.columns = df.columns.str.strip()

        if 'Conditions' not in df.columns:
            print("Missing required 'Conditions' column.")
            return pd.DataFrame()

        cancer_type = cancer_type.lower().strip()

        # Define broad match rules
        solid_tumors = ["breast", "lung", "colon", "pancreas", "prostate", "liver", "gallbladder", "kidney",
                        "ovary", "brain", "melanoma"]
        liquid_tumors = ["leukemia", "lymphoma", "myeloma", "aml", "cll", "cml", "b-cell", "t-cell"]
        broad_map = {
            "neoplasm": solid_tumors,
            "malignant": ["advanced", "metastatic", "stage iii", "stage iv"],
            "metastatic": ["metastatic", "advanced", "stage iii", "stage iv"],
            "advanced": ["advanced", "stage iii", "stage iv"],
            "hematologic": liquid_tumors,
            "liquid tumor": liquid_tumors,
            "solid tumor": solid_tumors,
            "endocrine": ["pancreas", "liver", "gallbladder"],
        }

        keywords = broad_map.get(cancer_type, [cancer_type])
        keywords = [kw.lower() for kw in keywords]

        def matches_condition(cond: str):
            cond = str(cond).lower()
            return any(kw in cond for kw in keywords)

        df_filtered = df[df['Conditions'].apply(matches_condition)]

        return df_filtered.reset_index(drop=True)

    except Exception as e:
        print("Error:", e)
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
    chat_bubble(user_input, sender="user")
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Increment turn count
    st.session_state.turn_count += 1

    # Extract patient information dynamically based on user input
    extract_patient_info(st.session_state.messages)

    # Determine if we should generate recommendations
    last_ai_message = next((msg for msg in reversed(st.session_state.messages)
                            if isinstance(msg, AIMessage)), None)

    final_question_indicator = last_ai_message and any(phrase in last_ai_message.content.lower()
                                                       for phrase in
                                                       ["final question", "last question", "one more question"])

    enough_turns = st.session_state.turn_count >= 10

    if final_question_indicator or enough_turns or st.session_state.questions_complete:
        st.session_state.questions_complete = True

        # Add a human-like thinking message
        chat_bubble(random.choice(get_thinking_phrases()), sender="assistant")

        # Final stage: Generate drug recommendations
        with st.spinner("Analyzing clinical information..."):
            # Extract the conversation for context
            conversation_context = ""
            for msg in st.session_state.messages:
                if isinstance(msg, HumanMessage):
                    conversation_context += f"User: {msg.content}\n"
                elif isinstance(msg, AIMessage) and not isinstance(msg, SystemMessage):
                    conversation_context += f"Doctor: {msg.content}\n"

            # Generate a comprehensive patient summary
            patient_info = extract_patient_info(st.session_state.messages)
            cancer_type = patient_info.get("cancer_type", "Unknown cancer type")

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
                patient_summary = chat_model.invoke([
                    SystemMessage(content="You are an oncologist creating a precise clinical assessment."),
                    HumanMessage(content=summary_prompt)
                ])

                # Explicitly add the cancer type to the summary for emphasis
                enhanced_summary = f"Patient has {cancer_type}. " + patient_summary.content

                # Load the drug recommendation chain (cached)
                drug_chain = load_system()

                # Use the summary to get drug recommendations
                recommendation = drug_chain.invoke(enhanced_summary)

                # Add a clinical introduction to recommendations
                empathetic_intro = random.choice(get_empathetic_intros())
                final_recommendation = f"{empathetic_intro}\n\n{recommendation}"

                trials_section = ""  # Initialize the trials section string
                trial_count = 1  # Initialize a counter for numbering the trials
                matching_trials = get_matching_trials(cancer_type)
                # Loop through the first 3 matching trials and add them to trials_section
                for _, row in matching_trials.head(3).iterrows():
                    trials_section += (
                         f"\n\n**Trial {trial_count}:**\n\n"
                        f"**Cancer Type**: {row.get('Conditions', 'N/A')}\n\n"
                        f"**Location**: {row.get('Locations', 'No description available.')}\n\n"
                        f"**Age**: {row.get('Age', 'N/A')}\n\n"
                        f"**Sex**: {row.get('Sex', 'N/A')}\n\n"
                        f"**Start Date**: {row.get('Start Date', 'N/A')}\n\n"
                        f"**Primary Completion Date**: {row.get('Primary Completion Date', 'N/A')}\n\n"
                        f"**Completion Date**: {row.get('Completion Date', 'N/A')}\n\n"
                        "\n---\n"
                    )
                    trial_count += 1

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
                chat_bubble(final_recommendation, sender="assistant")

                # Reset questions complete for future interactions
                st.session_state.questions_complete = True

            except Exception as e:
                error_message = (
                    f"I'm unable to formulate a complete treatment recommendation at this time. "
                    "This may be due to insufficient clinical information or the complexity of the case. "
                    "Could you provide additional details about the patient's disease characteristics or relevant biomarkers?"
                )
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append(AIMessage(content=error_message))
                chat_bubble(error_message, sender="assistant")


    else:
        # Continue asking context-aware questions
        with st.spinner("Reviewing information..."):
            if random.random() < 0.2:
                chat_bubble(random.choice(get_thinking_phrases()), sender="assistant")

            patient_info = extract_patient_info(st.session_state.messages)

            patient_context = ""
            if patient_info.get("age"):
                patient_context += f"Patient age: {patient_info['age']}. "
            if patient_info.get("gender"):
                patient_context += f"Patient gender: {patient_info['gender']}. "
            if patient_info.get("cancer_type"):
                patient_context += f"Cancer type: {patient_info['cancer_type']}. "
            if patient_info.get("stage"):
                patient_context += f"Disease stage: {patient_info['stage']}. "

            if patient_info.get("prior_treatments"):
                treatments = ", ".join(patient_info["prior_treatments"])
                patient_context += f"Prior treatments: {treatments}. "

            # Guidance message to generate context-aware questions
            guidance_msg = SystemMessage(
                content="""
            You are a compassionate clinical assistant helping a patient navigate their prostate cancer diagnosis.

            Your job is to collect **structured clinical information** by asking **one empathetic, clear question at a time** — based on what has *not* yet been shared. Do **not repeat or rephrase** previous questions if they’ve been declined.

            Follow this flow:

            1. **If not yet shared**, ask for: age, gender, diagnosis details (cancer type, stage, date of diagnosis).
            2. **If not yet confirmed**, ask if the cancer has spread (metastasis) — imaging, lymph nodes, or distant organs.
            3. **If not yet discussed**, ask if they’ve had **biomarker testing** (e.g., AR-V7, BRCA, PTEN for prostate cancer).
            4. Then ask about **treatment history** (any past or ongoing therapies).
            5. Then **imaging results** (shrinkage, stable, progressed).
            6. Then ask if they are interested in learning about **clinical trials or new therapeutic options**.
            7. Ask if they'd like to **receive more information** (email or phone).
            8. If they consent, collect **contact details**.

            Important:
            - **Never repeat** questions that the patient has declined to answer.
            - If they say "no" to a question, gently move on to the **next category**.
            - Always keep a **kind, human, and non-pushy tone**.
            """
            )

            # Generate next question with patient context
            next_question_msg = chat_model.invoke(st.session_state.messages + [guidance_msg])
            response_content = next_question_msg.content

            if len(response_content.strip()) < 10:
                # Fallback question if none is generated
                response_content = "Could you specify the type and location of the cancer?"

            # Avoid repetitive phrases like "Thank you for sharing"
            response_content = re.sub(r"thank you for sharing", "I appreciate your input", response_content)

            st.session_state.messages.append(AIMessage(content=response_content))
            chat_bubble(response_content, sender="assistant")

# Sidebar
st.sidebar.title("Oncology Consultation")
st.sidebar.markdown("---")

if st.sidebar.button("Start New Consultation"):
    st.session_state.messages = [initial_system_msg]
    # Add a welcome message
    welcome_message = AIMessage(
        content=(
            "Hi there, I’m designed to use AI to help you make better treatment decisions."
        "Kindly share you age, gender and diagnosis? (If you prefer not to share details that’s ok for now)"

    )
    )
    st.session_state.messages.append(welcome_message)
    st.session_state.turn_count = 0
    st.session_state.questions_complete = False
    st.session_state.patient_info = {}
    st.rerun()
