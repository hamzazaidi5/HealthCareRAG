from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from utils.config import Config


class DrugRecommendationChain:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatGroq(temperature=0, model_name=Config.LLM_MODEL)


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

        self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
        )

    def invoke(self, question):
        return self.chain.invoke(question)