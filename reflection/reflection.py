from typing import List, Dict, Optional
from base import BaseRouter, BaseRetriever, BaseReflection, BaseResponder, Message
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

class LangchainReflection(BaseReflection):
    def __init__(self, llm: ChatOpenAI):
        self.query_expansion_chain = LLMChain(
            llm=llm,
            prompt=ChatPromptTemplate.from_template("""
            Dựa vào lịch sử trò chuyện và câu hỏi hiện tại, 
            hãy tạo một câu truy vấn để tìm kiếm thông tin liên quan.
            Câu truy vấn nên bao gồm các từ khóa quan trọng và ngữ cảnh từ cuộc hội thoại.

            Lịch sử trò chuyện:
            {chat_history}

            Câu hỏi hiện tại: {question}

            Câu truy vấn mở rộng:""")
        )

    async def expand_query(self, question: str, chat_history: List[Message]) -> str:
        formatted_history = "\n".join([
            f"{'Người dùng' if msg.type == 'human' else 'Trợ lý'}: {msg.content}"
            for msg in chat_history[-5:]
        ])
        return await self.query_expansion_chain.arun(
            question=question,
            chat_history=formatted_history
        )