from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from utils.config import Config


class DrugRecommendationChain:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatGroq(temperature=0, model_name=Config.LLM_MODEL)

        # self.prompt = ChatPromptTemplate.from_template(
        #     """You are an oncology drug recommendation system. Use only the provided context to answer.
        #
        #     Context: {context}
        #
        #     Question: {question}
        #
        #     Provide a detailed response with:
        #     1. Drug suitability assessment
        #     2. Key efficacy metrics
        #     3. Safety considerations
        #     4. Relevant study findings
        #     If information is unavailable, state 'Data not available'."""
        # )

#         self.prompt = ChatPromptTemplate.from_template(
#             """You are an AI assistant providing evidence-based oncology treatment insights.
# Your responses must be grounded in clinical trial data from FDA-approved drugs and reference outcomes directly from the FDA.gov website.
#
# ### Key Considerations for Responses:
# 1️⃣ Overall Survival (OS) is the Gold Standard – if the OS benefit is marginal, indicate that the impact may be clinically insignificant.
# 2️⃣ Other outcomes such as PFS or ORR do not prove a survival benefit – note that improvements in these metrics do not necessarily extend a patient's life.
# 3️⃣ Many prescribed drugs are not proven in the patient’s specific cancer type – off-label prescribing indicates experimental use without solid survival data.
# 4️⃣ Be direct, clear, and factual by relaying only verified clinical data, referencing FDA clinical trials and regulatory decisions.
# 5️⃣ If the drug is present in the provided CSV data, always include the following caution line in your response:
#    **Be cautious**: Many treatments **do not improve survival** but are still widely used.
# 6️⃣ If the drug is not available in the CSV data, clearly state: "Drug is not available".
#
# ### Response Format:
# ✅ **Drug Name:** <Drug Name or "Drug is not available">
# 📊 **Clinical Trial Data:** FDA.gov
# ⏳ **Overall Survival (OS) Benefit:** <value or statement>
# ⚠️ **PFS Improvement Only (No OS Benefit):** <Yes/No>
# 🔬 **Off-Label Use in this Cancer Type?** <Yes/No>
# 💡 **Summary:**
# - <Detailed summary highlighting key points>
# - **Be cautious**: Many treatments **do not improve survival** but are still widely used.
#
# Context: {context}
#
# Question: {question}"""
#         )

        self.prompt = ChatPromptTemplate.from_template(
            """You are an AI assistant providing evidence-based oncology treatment insights.
Your responses must be grounded in clinical trial data from FDA-approved drugs and reference outcomes directly from the FDA.gov website.

### Key Instructions:
1️⃣ Overall Survival (OS) is the Gold Standard – if the OS benefit is marginal, indicate that the impact may be clinically insignificant.
2️⃣ Other outcomes such as PFS or ORR do not prove a survival benefit – clearly state that improvements in these metrics do not necessarily extend a patient's life.
3️⃣ Many prescribed drugs are not proven in the patient’s specific cancer type – off-label prescribing indicates experimental use without solid survival data.
4️⃣ Always output your answer in two parts:
    - **Key–Value Section:** Each line must have a key and its corresponding value (exactly as shown below):

      ✅ Drug Name: <Drug Name or "Drug is not available">
      📊 Clinical Trial Data: FDA.gov
      ⏳ Overall Survival (OS) Benefit: <value or statement or "Not applicable">
      ⚠️ PFS Improvement Only (No OS Benefit): <Yes/No or "Not applicable">
      🔬 Off-Label Use in this Cancer Type?: <Yes/No or "Not applicable">

    - **Summary Section:** Preceded by "💡 Summary:" on a new line, provide a detailed explanation.
5️⃣ If the queried drug is present in the CSV data, always append the following line at the end of the Summary:

   Be cautious: Many treatments do not improve survival but are still widely used.


### Response Format Example:
✅ Drug Name: Trastuzumab
📊 Clinical Trial Data: FDA.gov
⏳ Overall Survival (OS) Benefit: +3.2 months
⚠️ PFS Improvement Only (No OS Benefit): Yes
🔬 Off-Label Use in this Cancer Type?: Yes
💡 Summary:
- Trastuzumab has a modest OS improvement in HER2+ breast cancer but is not proven to extend survival in other cancer types.
- It is essential to verify clinical trial data.
*Be cautious*: Many treatments do not improve survival but are still widely used.

Context: {context}

Question: {question}"""
        )

        # self.prompt = ChatPromptTemplate.from_template(
        #     """You are an AI assistant providing evidence-based oncology treatment insights.
        # Your responses must be grounded in clinical trial data from FDA-approved drugs and reference outcomes directly from the FDA.gov website.
        #
        # ### Key Instructions:
        # 1️⃣ Overall Survival (OS) is the Gold Standard – if the OS benefit is marginal, indicate clinical insignificance.
        # 2️⃣ Other outcomes (PFS/ORR) don't prove survival – explicitly state they don't guarantee life extension.
        # 3️⃣ Flag off-label use as experimental when lacking cancer-specific data.
        # 4️⃣ Drug Check First:
        #    - If drug NOT in CSV data → ONLY show:
        #      ✅ Drug Name: Drug is not available
        #      (STOP HERE - no other sections)
        #    - If drug EXISTS → include ALL sections below
        #
        # 5️⃣ When drug exists, ALWAYS format as:
        #     - **Key–Value Section:**
        #       ✅ Drug Name: <Exact name from data>
        #       📊 Clinical Trial Data: FDA.gov
        #       ⏳ OS Benefit: <value or statement or "Not applicable">
        #       ⚠️ PFS Only: <Yes/No/Not applicable>
        #       🔬 Off-Label: <Yes/No/Not applicable>
        #
        #     - **Summary Section:**
        #       💡 Summary: <2-3 bullet points>
        #       Be cautious: Many treatments don't improve survival but are widely used.
        #
        # ### Response Examples:
        # ❌ Drug Not Found:
        # ✅ Drug Name: Drug is not available
        #
        # ✅ Drug Found:
        # ✅ Drug Name: Pembrolizumab
        # 📊 Clinical Trial Data: FDA.gov
        # ⏳ OS Benefit: +4.1 months
        # ⚠️ PFS Only: No
        # 🔬 Off-Label: No
        # 💡 Summary:
        # - Shows OS benefit in metastatic NSCLC with PD-L1 >50%
        # - First-line standard of care in this indication
        # - Be cautious: Many treatments don't improve survival but are widely used.
        #
        # Context: {context}
        # Question: {question}"""
        # )

        self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
        )

    def invoke(self, question):
        return self.chain.invoke(question)