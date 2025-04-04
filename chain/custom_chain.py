from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import re
from typing import List, Dict
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class DrugRecommendationChain:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

        # Enhanced prompt with stronger emphasis on OS as the gold standard while ensuring accuracy
        self.prompt = ChatPromptTemplate.from_template(
            """You are an AI assistant providing evidence-based oncology treatment insights.
        Your responses must be grounded in clinical trial data from FDA-approved drugs and reference outcomes directly from the FDA.gov website.

        ### CRITICAL INSTRUCTION:
        Ensure that all drug recommendations are SPECIFICALLY APPROVED for the EXACT cancer type mentioned in the patient information.
        PROVIDE A MAXIMUM OF 2 DRUG RECOMMENDATIONS ONLY, prioritizing those with the strongest overall survival benefit.
        If you cannot find FDA-approved drugs for the EXACT cancer type, state this clearly and recommend standard of care options.

        ### Patient Information:
        {question}

        ### Retrieved Context (use ONLY this data for recommendations):
        {context}

        ### ⚠️ OVERALL SURVIVAL (OS) IS THE GOLD STANDARD ⚠️
        - **EMPHASIZE THIS FACT IN YOUR RESPONSE**: OS benefit (helping patients live longer) is the most important outcome.
        - If survival data is provided in the context, use the actual numbers provided.
        - Only state "NO PROVEN SURVIVAL BENEFIT" if:
           a) The context explicitly states OS was not improved, OR
           b) The context only mentions PFS/ORR improvements without any OS data
        - If OS improvement exists but is minimal (1-3 months), clearly state: "Minimal survival benefit of only X months."
        - For each drug, specify whether there is evidence it helps patients live longer.

        ### Key Considerations for Responses:
        1️⃣ **Cancer Type Specificity**
           - ONLY recommend drugs that are FDA-approved for the EXACT CANCER TYPE mentioned in the patient information.
           - If no drugs in the context are approved for this cancer type, clearly state this.

        2️⃣ **Rank Recommendations by Survival Benefit**
           - Drugs with statistically significant OS improvement should be listed first.
           - For each drug, prominently display OS benefit in months/years when data is available.
           - If OS data is not provided in context, state "OS data not available in current information" rather than claiming no benefit.

        3️⃣ **Other Outcomes (PFS, ORR) Are Secondary**
           - Clearly label: "⚠️ IMPORTANT: Progression-Free Survival (PFS) improvements alone do NOT necessarily mean patients will live longer."
           - Explain that PFS and response rates are surrogate endpoints that may not translate to actual survival benefits.

        4️⃣ **Be Direct, Clear, and Factual**
           - Never invent survival data not present in the context.
           - Use precise language about survival benefits based on the data provided.
           - If the context lacks OS data for a drug, acknowledge this gap rather than making claims either way.

        ### Response Format:
        - **Introduction:** Begin with "Based on the patient information provided, here are the FDA-approved drugs for [EXACT CANCER TYPE] with a focus on actual survival benefits:"

        - **Drug Recommendations:** For each recommended drug, include:
           - **Drug Name**
           - **❗ Survival Impact:** [One of: "Extends life by X months/years" OR "NO PROVEN SURVIVAL BENEFIT" OR "OS data not available in current information"]
           - **FDA Approval Status** for this specific cancer type
           - **Clinical Trial Data** (Source: FDA.gov) - include actual numbers from context
           - **Other Outcomes** (PFS, ORR) with clear indication these are not survival benefits
           - **Off-Label Use?** (Yes/No for this cancer type)

        - **Summary:** Prefixed with "💡 **SUMMARY:**" that emphasizes:
           1. Whether any recommended drugs have proven OS benefits (based solely on provided context)
           2. The magnitude of any survival benefit (in months/years)
           3. Clear statement if OS data is missing from the context

        - **Final Caution Note:** End with "**⚠️ IMPORTANT REMINDER:** Some treatments may not improve survival but are still commonly used."
        """
        )

        # Build the chain: retrieve => fill prompt => run LLM => parse
        self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
        )

    def _extract_cancer_type(self, question):
        """Extract the cancer type from the patient information"""
        # Simple regex pattern to find cancer types
        cancer_patterns = [
            r"(?i)diagnosed with\s+([^.,;]+(?:\s+cancer|\s+carcinoma|\s+sarcoma|\s+lymphoma|\s+leukemia|\s+melanoma|\s+tumor|\s+neoplasm|\s+myeloma))",
            r"(?i)has\s+([^.,;]+(?:\s+cancer|\s+carcinoma|\s+sarcoma|\s+lymphoma|\s+leukemia|\s+melanoma|\s+tumor|\s+neoplasm|\s+myeloma))",
            r"(?i)patient with\s+([^.,;]+(?:\s+cancer|\s+carcinoma|\s+sarcoma|\s+lymphoma|\s+leukemia|\s+melanoma|\s+tumor|\s+neoplasm|\s+myeloma))",
            r"(?i)patient has\s+([^.,;]+(?:\s+cancer|\s+carcinoma|\s+sarcoma|\s+lymphoma|\s+leukemia|\s+melanoma|\s+tumor|\s+neoplasm|\s+myeloma))"
        ]

        for pattern in cancer_patterns:
            match = re.search(pattern, question)
            if match:
                return match.group(1).strip()

        # If no match found, use NLP to extract it
        try:
            extraction_prompt = "Extract only the cancer type from this text. Reply with just the cancer type name, nothing else: " + question
            cancer_type = self.llm.invoke(extraction_prompt)
            return str(cancer_type).strip()
        except:
            return "unknown cancer type"

    def _enhance_question_with_os_focus(self, question, cancer_type):
        """Add explicit OS focus to the patient question"""
        enhanced = f"The patient has {cancer_type}. " + question
        if "overall survival" not in enhanced.lower() and "os" not in enhanced.lower():
            enhanced += " Please prioritize information about OVERALL SURVIVAL benefits and clearly distinguish between drugs that help patients live longer versus those that only improve disease metrics."
        return enhanced

    def invoke(self, question: str) -> str:
        try:
            # Extract the cancer type for enhanced precision
            cancer_type = self._extract_cancer_type(question)

            # Enhance the question with explicit cancer type and OS focus
            enhanced_question = self._enhance_question_with_os_focus(question, cancer_type)

            # Get results with the enhanced question
            result = self.chain.invoke(enhanced_question)

            # If we got an empty result, return a helpful message
            if not result or result.strip() == "":
                return f"""Based on the information provided, I cannot find specific FDA-approved drugs for {cancer_type} in my knowledge base. 

⚠️ IMPORTANT REMINDER: When evaluating cancer treatments, overall survival (helping patients live longer) is the gold standard outcome.

This could be due to:
1. {cancer_type} may be rare or have specialized treatment protocols
2. The database may not contain the latest FDA approvals for this specific cancer type
3. Treatment may be based on NCCN guidelines rather than specific FDA-approved drugs

Please consult with a medical oncologist who can provide personalized treatment recommendations based on the latest clinical guidelines and discuss which treatments, if any, have been proven to extend life."""

            return result
        except Exception as e:
            print(f"Error in DrugRecommendationChain: {str(e)}")
            return f"""I apologize, but I encountered an error while generating drug recommendations for this cancer type. 

Please verify that:
1. The patient information clearly specifies the cancer type, stage, and relevant medical history
2. Your oncology database contains FDA-approved drugs for this condition

Technical details (for developers): {str(e)}"""


class ClinicalTrialRecommendationChain:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

        self.trial_recommendation_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an AI assistant providing evidence-based clinical trial recommendations.
            Your responses must be grounded in the provided trial data and focus on currently recruiting trials.

            ### CRITICAL INSTRUCTION:
            Ensure that all trial recommendations are SPECIFICALLY RELEVANT to the patient's:
            1. Cancer type and stage
            2. Location
            3. Previous treatment history
            4. Biomarker status (if provided)

            ### Patient Information:
            {question}

            ### Retrieved Trials (use ONLY this data for recommendations):
            {context}

            ### Key Considerations for Trial Selection:
            1️⃣ **Trial Relevance**
               - Prioritize trials specifically targeting the patient's cancer type and mutations
               - Consider previous treatment history in eligibility
               - Focus on trials in the patient's location or nearby

            2️⃣ **Trial Phase and Outcomes**
               - Prioritize trials measuring overall survival when available
               - Clearly indicate the phase of each trial
               - Highlight innovative treatment approaches

            3️⃣ **Patient Eligibility**
               - List key inclusion/exclusion criteria
               - Note any specific biomarker requirements
               - Mention performance status requirements

            ### Response Format:
            - **Introduction:** Begin with "Based on your clinical profile, here are the most relevant clinical trials currently recruiting:"

            - **Trial Recommendations:** For each trial (maximum 3), include:
               - **Trial ID and Title**
               - **Location and Site Details**
               - **Key Eligibility Criteria**
               - **Treatment Approach**
               - **Phase and Primary Outcomes**
               - **Next Steps for Enrollment**

            - **Summary:** Prefixed with "💡 **KEY POINTS:**"
               1. Why these trials were selected for you
               2. Any specific requirements to note
               3. Immediate next steps

            - **Final Note:** End with "⚠️ IMPORTANT: Please discuss these trial options with your healthcare team to determine the most appropriate choice for your specific situation."

            Keep the response focused and actionable, prioritizing trials that best match the patient's profile."""
        )

        self.chain = LLMChain(llm=llm, prompt=self.trial_recommendation_prompt)

    def invoke(self, question: str) -> str:
        try:
            # Get relevant trials
            docs = self.retriever.get_relevant_documents(question)[:3]

            # Create concise context
            context = "\n".join(f"Trial {i + 1}: {doc.page_content[:300]}..."
                                for i, doc in enumerate(docs))

            # Generate recommendations
            response = self.chain.run(
                context=context,
                question=question
            )

            if not response or response.strip() == "":
                return f"""Based on the provided information, I cannot find any currently recruiting clinical trials that match your specific criteria.

⚠️ IMPORTANT REMINDER: Clinical trial availability changes frequently.

This could be due to:
1. Very specific cancer type or mutation requirements
2. Location constraints
3. Current trial recruitment status
4. Specific eligibility criteria

Next Steps:
1. Discuss with your oncologist about other trial options
2. Check clinicaltrials.gov for updated listings
3. Consider expanding your search radius for more options"""

            return response

        except Exception as e:
            return (
                "I apologize, but I'm having trouble processing the clinical trials data. "
                "Please verify:\n"
                "1. Your location is clearly specified\n"
                "2. Cancer type and stage are provided\n"
                "3. Any relevant biomarkers or previous treatments are mentioned\n\n"
                "This will help find the most appropriate trial matches for your situation."
            )


class ComprehensiveRecommendationChain:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

        self.recommendation_prompt = PromptTemplate(
            input_variables=["trials_context", "oncology_context", "question"],
            template="""You are an AI assistant providing evidence-based oncology treatment insights.
            Your responses must be grounded in clinical trial data and FDA-approved treatments.

            ### Patient Information:
            {question}

            ### Available Clinical Trials:
            {trials_context}

            ### FDA-Approved Treatments:
            {oncology_context}

            ### CRITICAL INSTRUCTION:
            Ensure recommendations are SPECIFICALLY RELEVANT to the patient's:
            1. Cancer type and stage
            2. Location
            3. Previous treatment history
            4. Biomarker status (if provided)

            ### Response Format:

            **PART 1: FDA-APPROVED TREATMENT OPTIONS**
            - List up to 2 most relevant FDA-approved treatments
            For each treatment:
            * Drug Name
            * ❗ Survival Impact: [Extends life by X months/years OR NO PROVEN SURVIVAL BENEFIT]
            * FDA Approval Status
            * Key Clinical Outcomes
            * Treatment Considerations

            **PART 2: RELEVANT CLINICAL TRIALS**
            - List up to 3 most relevant recruiting trials
            For each trial:
            * Trial ID and Title
            * Location and Site Details
            * Key Eligibility Criteria
            * Treatment Approach
            * Phase and Primary Outcomes
            * Next Steps for Enrollment

            💡 **SUMMARY:**
            1. Best current FDA-approved options
            2. Most promising trial options
            3. Suggested next steps

            ⚠️ **IMPORTANT REMINDER:** Please discuss these options with your healthcare team to determine the most appropriate treatment path for your specific situation."""
        )

        self.chain = LLMChain(llm=llm, prompt=self.recommendation_prompt)

    def invoke(self, question: str) -> str:
        try:
            # Get relevant documents
            docs = self.retriever.get_relevant_documents(question)

            if not docs:
                return (
                    "I apologize, but I couldn't find any relevant clinical trials or treatments "
                    "matching your criteria. This could be because:\n"
                    "1. The cancer type or location may need to be more specific\n"
                    "2. There might not be any active trials in your area\n"
                    "3. The data might need to be updated\n\n"
                    "Please try:\n"
                    "1. Specifying your exact cancer type and stage\n"
                    "2. Providing your preferred location for trials\n"
                    "3. Including any relevant biomarkers or previous treatments"
                )

            # Separate trials and oncology data
            trials_docs = [doc for doc in docs if doc.metadata.get('source') == 'clinical_trial']
            oncology_docs = [doc for doc in docs if doc.metadata.get('source') == 'oncology_data']

            # Format contexts (use empty string if no documents found)
            trials_context = "\n".join(
                doc.page_content for doc in trials_docs[:3]) if trials_docs else "No matching clinical trials found."
            oncology_context = "\n".join(doc.page_content for doc in oncology_docs[
                                                                     :2]) if oncology_docs else "No matching FDA-approved treatments found."

            # Generate recommendations
            response = self.chain.run(
                trials_context=trials_context,
                oncology_context=oncology_context,
                question=question
            )

            return response

        except Exception as e:
            print(f"Error in recommendation chain: {str(e)}")
            return (
                "I apologize, but I'm having trouble processing the available data. "
                "Please ensure you've provided:\n"
                "1. Your specific cancer type and stage\n"
                "2. Location for trial matching\n"
                "3. Previous treatment history\n"
                "4. Any relevant biomarkers\n\n"
                "This will help provide comprehensive treatment recommendations."
            )