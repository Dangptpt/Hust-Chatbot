from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from base import BaseRouter, Message
from langchain_core.output_parsers import StrOutputParser
from typing import List
from langchain_core.messages import HumanMessage, AIMessage

router_prompt = '''Từ lịch sử đoạn chat và câu hỏi của người dùng Hãy phân loại câu hỏi của người dùng vào một trong hai loại:
1. chitchat: Câu hỏi thông thường, chào hỏi, cảm ơn, thời tiết, làm thơ v.v.
2. rag: Câu hỏi cần tra cứu thông tin về quy chế, học bổng, đào tạo, giấy tờ, tuyển sinh, phí dịch vụ, hướng dẫn, thắc mắc,v.v

Câu hỏi: {question}
Phân loại (chỉ trả về "chitchat" hoặc "rag"):'''

class LangchainRouter(BaseRouter):
    def __init__(self, llm: ChatOpenAI):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là router phân loại câu hỏi cho chatbot của trường Đại học Bách khoa"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", router_prompt)
        ])
        
        self.chain = prompt | llm | StrOutputParser()

    async def classify(self, question: str, chat_history: List[Message], num_context: int = 2) -> str:
        formatted_history = [
            HumanMessage(content=msg.content) if msg.type == 'human'
            else AIMessage(content=msg.content)
            for msg in chat_history[-num_context*2:]
        ]
        response = await self.chain.ainvoke({
            "question": question,
            "chat_history": formatted_history
            })
        return response.strip().lower()