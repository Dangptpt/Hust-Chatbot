from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from base import BaseRouter

class LangchainRouter(BaseRouter):
    def __init__(self, llm: ChatOpenAI):
        self.router_chain = LLMChain(
            llm=llm,
            prompt=ChatPromptTemplate.from_template("""
            Bạn là router phân loại câu hỏi cho chatbot của trường Đại học Bách khoa.
            Hãy phân loại câu hỏi của người dùng vào một trong hai loại:
            1. chitchat: Câu hỏi thông thường, chào hỏi, cảm ơn
            2. rag: Câu hỏi cần tra cứu thông tin về quy chế, học bổng, đào tạo, giấy tờ, tuyển sinh

            Câu hỏi: {question}
            Phân loại (chỉ trả về "chitchat" hoặc "rag"):""")
        )

    async def classify(self, question: str) -> str:
        return (await self.router_chain.arun(question=question)).strip().lower()
