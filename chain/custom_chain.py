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
            """You are an oncology drug recommendation system. Use only the provided context to answer.

            Context: {context}

            Question: {question}

            Provide a detailed response with:
            1. Drug suitability assessment
            2. Key efficacy metrics
            3. Safety considerations
            4. Relevant study findings
            If information is unavailable, state 'Data not available'."""
        )

        self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
        )

    def invoke(self, question):
        return self.chain.invoke(question)