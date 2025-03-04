from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import re


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