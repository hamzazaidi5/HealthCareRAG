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

        # Enhanced prompt that emphasizes cancer type matching and relevance
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

        ### Key Considerations for Responses:
        1️⃣ **Cancer Type Specificity**
           - ONLY recommend drugs that are FDA-approved for the EXACT CANCER TYPE mentioned in the patient information.
           - If no drugs in the context are approved for this cancer type, clearly state this.

        2️⃣ **Overall Survival (OS) is the Gold Standard**
           - The most meaningful outcome measure is **OS (Overall Survival)**—whether a drug **helps patients live longer**.
           - If the OS benefit is marginal (e.g., only weeks/months), clarify that the impact may be clinically insignificant.

        3️⃣ **Other Outcomes (PFS, ORR) Do NOT Prove Survival Benefit**
           - Drugs approved based on **Progression-Free Survival (PFS)** or **Response Rate (ORR, DOR, CR, PR)** do not necessarily extend life.
           - Explicitly state: **A drug that improves PFS does not necessarily extend a patient's life.**
           - If a drug is approved solely on PFS or ORR without an OS benefit, highlight this fact.

        4️⃣ **Be Direct, Clear, and Factual**
           - Clearly indicate if a drug does not improve survival.
           - Avoid providing false hope or exaggerated benefits—only relay clinical data.
           - Always reference **FDA clinical trials and regulatory decisions**.

        ### Response Format:
        - **Introduction:** Begin with "Based on the patient information provided, here are the FDA-approved drugs and relevant survival data for [EXACT CANCER TYPE]:"

        - **Drug Recommendations:** For each recommended drug, include:
           - **Drug Name**
           - **FDA Approval Status** for this specific cancer type
           - **Clinical Trial Data** (Source: FDA.gov)
           - **Overall Survival (OS) Benefit** (specify months/years if available)
           - **PFS Improvement** (mention if no OS benefit)
           - **Off-Label Use?** (Yes/No for this cancer type)

        - **Summary:** Prefixed with "💡 **Summary:**" that concisely explains the recommendation, including any caveats regarding marginal benefits or off-label use.

        - **Final Caution Note:** End with "**Be cautious**: Some treatments may not improve survival but are still commonly used."
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

    def invoke(self, question: str) -> str:
        try:
            # Extract the cancer type for enhanced precision
            cancer_type = self._extract_cancer_type(question)

            # Enhance the question with explicit cancer type
            enhanced_question = f"The patient has {cancer_type}. " + question

            # Get results with the enhanced question
            result = self.chain.invoke(enhanced_question)

            # If we got an empty result, return a helpful message
            if not result or result.strip() == "":
                return f"""Based on the information provided, I cannot find specific FDA-approved drugs for {cancer_type} in my knowledge base. 

This could be due to:
1. {cancer_type} may be rare or have specialized treatment protocols
2. The database may not contain the latest FDA approvals for this specific cancer type
3. Treatment may be based on NCCN guidelines rather than specific FDA-approved drugs

Please consult with a medical oncologist who can provide personalized treatment recommendations based on the latest clinical guidelines."""

            return result
        except Exception as e:
            print(f"Error in DrugRecommendationChain: {str(e)}")
            return f"""I apologize, but I encountered an error while generating drug recommendations for this cancer type. 

Please verify that:
1. The patient information clearly specifies the cancer type, stage, and relevant medical history
2. Your oncology database contains FDA-approved drugs for this condition

Technical details (for developers): {str(e)}"""