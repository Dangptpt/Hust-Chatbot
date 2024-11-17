from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from base import BaseRouter
from base import BaseRetriever, BaseResponder, Message
from typing import List

class ChitchatResponder(BaseResponder):
    def __init__(self, llm: ChatOpenAI):
        self.chain = LLMChain(
            llm=llm,
            prompt=ChatPromptTemplate.from_template("""
            Hãy trả lời thân thiện và ngắn gọn.

            Lịch sử trò chuyện:
            {chat_history}

            Câu hỏi: {question}
            Trả lời:""")
        )

    async def respond(self, question: str, chat_history: List[Message], **kwargs) -> str:
        formatted_history = "\n".join([
            f"{'Người dùng' if msg.type == 'human' else 'Trợ lý'}: {msg.content}"
            for msg in chat_history[-5:]
        ])
        return await self.chain.arun(
            question=question,
            chat_history=formatted_history
        )

class RAGResponder(BaseResponder):
    def __init__(self, llm: ChatOpenAI):
        self.chain = LLMChain(
            llm=llm,
            prompt=ChatPromptTemplate.from_template("""
            Bạn là trợ lý ảo của trường Đại học Bách khoa.
            Dựa vào thông tin được cung cấp, hãy trả lời câu hỏi của người dùng
            Nếu không có đủ thông tin, hãy trả lời rằng bạn không biết

            Thông tin tham khảo:
            {context}

            Lịch sử trò chuyện:
            {chat_history}

            Câu hỏi: {question}
            Trả lời:""")
        )

    async def respond(self, question: str, chat_history: List[Message], context: str = "") -> str:
        formatted_history = "\n".join([
            f"{'Người dùng' if msg.type == 'human' else 'Trợ lý'}: {msg.content}"
            for msg in chat_history[-5:]
        ])
        return await self.chain.arun(
            question=question,
            chat_history=formatted_history,
            context=context
        )
