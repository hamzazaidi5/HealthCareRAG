import streamlit as st
from utils.config import Config
from utils.data_loader import load_csv_as_documents
from chain.custom_chain import DrugRecommendationChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


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


# Use Streamlit caching to load the system only once
@st.cache_resource
def load_system():
    retriever = initialize_system()
    drug_recommendation_chain = DrugRecommendationChain(retriever)
    return drug_recommendation_chain


def get_default_drug_for_cancer(cancer_type):
    """Return default drugs to query based on cancer type"""
    cancer_type = cancer_type.lower()

    if "breast" in cancer_type:
        if "her2+" in cancer_type or "her2 positive" in cancer_type:
            return "Trastuzumab"
        elif "her2-" in cancer_type or "her2 negative" in cancer_type:
            return "Tamoxifen"
        else:
            return "Tamoxifen, Trastuzumab"
    elif "lung" in cancer_type:
        return "Pembrolizumab"
    elif "colon" in cancer_type or "colorectal" in cancer_type:
        return "Fluorouracil"
    elif "prostate" in cancer_type:
        return "Abiraterone"
    elif "melanoma" in cancer_type:
        return "Nivolumab"
    elif "leukemia" in cancer_type:
        return "Imatinib"
    else:
        # Generic cancer drugs that are commonly used
        return "Cyclophosphamide, Doxorubicin, Paclitaxel"


def main():
    st.title("Oncology Treatment Information Assistant")

    # Initialize session state for conversation flow and data collection
    if "stage" not in st.session_state:
        st.session_state.stage = 0

    if "patient_data" not in st.session_state:
        st.session_state.patient_data = {
            "demographics": {},
            "cancer_info": {},
            "treatment_history": [],
            "considered_treatments": [],
            "recommendations_generated": False
        }

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Load recommendation system
    drug_chain = load_system()

    # Display conversation history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**Patient:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")

    # Initial welcome message
    if st.session_state.stage == 0:
        welcome_msg = "Hi there, I am a research specialist designed to give information on treatment options based on published data. Can you start by telling me a little bit about yourself? Age, gender, and relevant health history?"
        st.session_state.chat_history.append({"role": "assistant", "content": welcome_msg})
        st.session_state.stage = 1
        st.rerun()

    # Handle user input based on conversation stage
    with st.form(key="patient_form", clear_on_submit=True):
        user_input = st.text_input("Your response:", key="user_message")
        submit_button = st.form_submit_button("Send")

    if submit_button and user_input:
        # Add user response to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Process input based on current stage
        if st.session_state.stage == 1:  # Demographics
            st.session_state.patient_data["demographics"]["response"] = user_input

            # Ask about cancer type
            next_question = "Thank you for sharing. What type of cancer have you been diagnosed with?"
            st.session_state.chat_history.append({"role": "assistant", "content": next_question})
            st.session_state.stage = 2

        elif st.session_state.stage == 2:  # Cancer type
            st.session_state.patient_data["cancer_info"]["type"] = user_input

            # Ask about cancer spread/stage
            next_question = "How much has the cancer spread? Is it in the lymph nodes or other systems? Do you know what stage it is?"
            st.session_state.chat_history.append({"role": "assistant", "content": next_question})
            st.session_state.stage = 3

        elif st.session_state.stage == 3:  # Cancer spread/stage
            st.session_state.patient_data["cancer_info"]["stage"] = user_input

            # Ask about cancer subtype
            next_question = "Do you know the subtype of your cancer or any genetic expressions that have been identified (like HER2, EGFR, PD-L1, etc.)?"
            st.session_state.chat_history.append({"role": "assistant", "content": next_question})
            st.session_state.stage = 4

        elif st.session_state.stage == 4:  # Cancer subtype
            st.session_state.patient_data["cancer_info"]["subtype"] = user_input

            # Ask about treatment history
            next_question = "What treatments have you had so far for your cancer?"
            st.session_state.chat_history.append({"role": "assistant", "content": next_question})
            st.session_state.stage = 5

        elif st.session_state.stage == 5:  # Treatment history
            st.session_state.patient_data["treatment_history"] = user_input

            # Ask about treatments being considered
            next_question = "What specific medications or drugs is your doctor recommending or are you considering? Please provide the names if possible."
            st.session_state.chat_history.append({"role": "assistant", "content": next_question})
            st.session_state.stage = 6

        elif st.session_state.stage == 6:  # Treatments being considered
            st.session_state.patient_data["considered_treatments"] = user_input

            # Summarize collected information
            summary = f"""
            Thank you for providing this information. Based on what you've shared:

            Demographics: {st.session_state.patient_data["demographics"]["response"]}
            Cancer Type: {st.session_state.patient_data["cancer_info"]["type"]}
            Cancer Stage: {st.session_state.patient_data["cancer_info"]["stage"]}
            Cancer Subtype: {st.session_state.patient_data["cancer_info"]["subtype"]}
            Prior Treatments: {st.session_state.patient_data["treatment_history"]}
            Considered Treatments: {st.session_state.patient_data["considered_treatments"]}

            I'll now search for evidence-based treatment options that may be relevant to your situation.
            """

            st.session_state.chat_history.append({"role": "assistant", "content": summary})

            # Move to the next stage where we'll generate drug recommendations
            st.session_state.stage = 7
            st.rerun()

        elif st.session_state.stage == 7:  # Generate recommendations
            if not st.session_state.patient_data.get("recommendations_generated", False):
                # Determine what drugs to query
                considered_treatments = st.session_state.patient_data["considered_treatments"].lower()
                cancer_type = st.session_state.patient_data["cancer_info"]["type"]

                # If user input is vague, use default drugs based on cancer type
                if considered_treatments in ["any drug", "drugs", "medication", "any", "not sure"]:
                    drugs_to_query = get_default_drug_for_cancer(cancer_type)
                else:
                    drugs_to_query = st.session_state.patient_data["considered_treatments"]

                # Prepare and send query to drug recommendation chain
                query = f"Provide information about {drugs_to_query} for {cancer_type}"

                try:
                    # Generate drug recommendation
                    drug_info = drug_chain.invoke(query)

                    # Display the drug information
                    st.session_state.chat_history.append({"role": "assistant", "content": drug_info})

                    # Mark recommendations as generated
                    st.session_state.patient_data["recommendations_generated"] = True

                    # Ask if they need more information
                    follow_up = "Do you have any questions about this information or would you like to know about a different treatment?"
                    st.session_state.chat_history.append({"role": "assistant", "content": follow_up})

                    # Move to follow-up stage
                    st.session_state.stage = 8
                except Exception as e:
                    # Handle errors and provide fallback response
                    fallback_response = f"""
                    ✅ Drug Name: {drugs_to_query}
                    📊 Clinical Trial Data: FDA.gov
                    ⏳ Overall Survival (OS) Benefit: Data not available
                    ⚠️ PFS Improvement Only (No OS Benefit): Not applicable
                    🔬 Off-Label Use in this Cancer Type?: Not applicable

                    💡 Summary:
                    - Limited data is available in our database for {drugs_to_query} in {cancer_type}.
                    - It's important to discuss the complete clinical evidence with your oncologist.
                    - Be cautious: Many treatments do not improve survival but are still widely used.
                    """

                    st.session_state.chat_history.append({"role": "assistant", "content": fallback_response})

                    # Mark recommendations as generated
                    st.session_state.patient_data["recommendations_generated"] = True

                    # Ask if they need more information
                    follow_up = "Would you like information about a different treatment option?"
                    st.session_state.chat_history.append({"role": "assistant", "content": follow_up})

                    # Move to follow-up stage
                    st.session_state.stage = 8

                st.rerun()
            else:
                # Process follow-up question
                follow_up_query = f"Patient asking about {st.session_state.patient_data['considered_treatments']} for {st.session_state.patient_data['cancer_info']['type']}: {user_input}"

                try:
                    follow_up_response = drug_chain.invoke(follow_up_query)
                    st.session_state.chat_history.append({"role": "assistant", "content": follow_up_response})
                except Exception as e:
                    error_msg = "I apologize, but I couldn't find specific information to answer your question. Would you like to ask about a different medication?"
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

        elif st.session_state.stage == 8:  # Follow-up questions
            # Process follow-up questions
            follow_up_query = f"Patient asking about {st.session_state.patient_data['considered_treatments']} for {st.session_state.patient_data['cancer_info']['type']}: {user_input}"

            try:
                follow_up_response = drug_chain.invoke(follow_up_query)
                st.session_state.chat_history.append({"role": "assistant", "content": follow_up_response})
            except Exception as e:
                error_msg = "I apologize, but I couldn't find specific information to answer your question. Would you like to ask about a different medication?"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

        st.rerun()

    # Handle automatic drug recommendation generation
    if st.session_state.stage == 7 and not st.session_state.patient_data.get("recommendations_generated", False):
        # This ensures recommendations are generated automatically
        considered_treatments = st.session_state.patient_data["considered_treatments"].lower()
        cancer_type = st.session_state.patient_data["cancer_info"]["type"]

        # If user input is vague, use default drugs based on cancer type
        if considered_treatments in ["any drug", "drugs", "medication", "any", "not sure"]:
            drugs_to_query = get_default_drug_for_cancer(cancer_type)
        else:
            drugs_to_query = st.session_state.patient_data["considered_treatments"]

        # Prepare and send query to drug recommendation chain
        query = f"Provide information about {drugs_to_query} for {cancer_type}"

        try:
            # Generate drug recommendation
            drug_info = drug_chain.invoke(query)

            # Display the drug information
            st.session_state.chat_history.append({"role": "assistant", "content": drug_info})

            # Mark recommendations as generated
            st.session_state.patient_data["recommendations_generated"] = True

            # Ask if they need more information
            follow_up = "Do you have any questions about this information or would you like to know about a different treatment?"
            st.session_state.chat_history.append({"role": "assistant", "content": follow_up})

            # Move to follow-up stage
            st.session_state.stage = 8
        except Exception as e:
            # Handle errors and provide fallback response
            fallback_response = f"""
            ✅ Drug Name: {drugs_to_query}
            📊 Clinical Trial Data: FDA.gov
            ⏳ Overall Survival (OS) Benefit: Data not available
            ⚠️ PFS Improvement Only (No OS Benefit): Not applicable
            🔬 Off-Label Use in this Cancer Type?: Not applicable

            💡 Summary:
            - Limited data is available in our database for {drugs_to_query} in {cancer_type}.
            - It's important to discuss the complete clinical evidence with your oncologist.
            - Be cautious: Many treatments do not improve survival but are still widely used.
            """

            st.session_state.chat_history.append({"role": "assistant", "content": fallback_response})

            # Mark recommendations as generated
            st.session_state.patient_data["recommendations_generated"] = True

            # Ask if they need more information
            follow_up = "Would you like information about a different treatment option?"
            st.session_state.chat_history.append({"role": "assistant", "content": follow_up})

            # Move to follow-up stage
            st.session_state.stage = 8

        st.rerun()

    # Option to start a new conversation
    if st.session_state.stage > 0 and st.button("Start New Conversation"):
        st.session_state.stage = 0
        st.session_state.patient_data = {
            "demographics": {},
            "cancer_info": {},
            "treatment_history": [],
            "considered_treatments": [],
            "recommendations_generated": False
        }
        st.session_state.chat_history = []
        st.rerun()


if __name__ == "__main__":
    main()