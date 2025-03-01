from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


class DrugRecommendationChain:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_template(
            """You are an AI assistant providing evidence-based oncology treatment insights.
        Your responses must be grounded in clinical trial data from FDA-approved drugs and reference outcomes directly from the FDA.gov website.

        ### Key Considerations for Responses:
        1️⃣ **Overall Survival (OS) is the Gold Standard**
           - The most meaningful outcome measure is **OS (Overall Survival)**—whether a drug **helps patients live longer**.
           - If the OS benefit is marginal (e.g., only weeks/months), clarify that the impact may be clinically insignificant.

        2️⃣ **Other Outcomes (PFS, ORR) Do NOT Prove Survival Benefit**
           - Drugs approved based on **Progression-Free Survival (PFS)** or **Response Rate (ORR, DOR, CR, PR)** do not necessarily extend life.
           - Explicitly state: **A drug that improves PFS does not necessarily extend a patient's life.**
           - If a drug is approved solely on PFS or ORR without an OS benefit, highlight this fact.

        3️⃣ **Many Prescribed Drugs Are NOT Proven in the Patient's Cancer Type**
           - Many drugs are prescribed off-label without robust evidence in the patient's specific cancer type.
           - Explain that **off-label prescribing** is experimental and may lack solid survival data.
           - Indicate whether the evidence in that cancer type is strong or weak.

        4️⃣ **Be Direct, Clear, and Factual**
           - Clearly indicate if a drug does not improve survival.
           - Avoid providing false hope or exaggerated benefits—only relay clinical data.
           - Always reference **FDA clinical trials and regulatory decisions**.

        ### Response Format:
        - **Introduction:** Begin with "Based on the patient information provided, here are the FDA-approved drugs and relevant survival data:"

        - **Drug Recommendations:** For each recommended drug, include:
           - **Drug Name**
           - **FDA Approval Status** for this specific cancer type
           - **Clinical Trial Data** (Source: FDA.gov)
           - **Overall Survival (OS) Benefit** (specify months/years if available)
           - **PFS Improvement** (mention if no OS benefit)
           - **Off-Label Use?** (Yes/No for this cancer type)

        - **Summary:** Prefixed with "💡 **Summary:**" that concisely explains the recommendation, including any caveats regarding marginal benefits or off-label use.

        - **Final Caution Note:** End with "**Be cautious**: Some treatments may not improve survival but are still commonly used."

        Remember to use the most recent FDA-approved data available in your knowledge base.

        Patient Information: {question}

        Retrieved Context (use this data to inform your recommendations):
        {context}
        """
        )

        # Build the chain: retrieve => fill prompt => run LLM => parse
        self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
        )

    def invoke(self, question: str) -> str:
        try:
            # Print debugging info
            print(f"Received question: {question}")

            # Get the results from the chain
            result = self.chain.invoke(question)

            # If we got an empty result, return a helpful message
            if not result or result.strip() == "":
                return """Based on the information provided, I cannot find specific FDA-approved drugs for this patient's condition in my knowledge base. 

This could be due to:
1. The cancer type may be rare or specialized
2. Insufficient details in the patient description
3. Limited data in my current knowledge base

Please consult with a medical oncologist who can provide personalized treatment recommendations based on the latest clinical guidelines and FDA approvals."""

            return result
        except Exception as e:
            print(f"Error in DrugRecommendationChain: {str(e)}")
            return f"""I apologize, but I encountered an error while generating drug recommendations. 

Please verify that:
1. The patient information includes the cancer type, stage, and relevant medical history
2. Your oncology database contains FDA-approved drugs for this condition

Technical details (for developers): {str(e)}"""