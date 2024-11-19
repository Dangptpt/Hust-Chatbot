from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from base import BaseRouter, BaseRetriever, BaseResponder, Message
from typing import List
from langchain_core.messages import HumanMessage, AIMessage


class ChitchatResponder(BaseResponder):
    def __init__(self, llm: ChatOpenAI):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là trợ lý ảo của trường Đại học Bách khoa Hà Nội. Hãy trả lời thân thiện và ngắn gọn"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        self.chain = prompt | llm | StrOutputParser()

    async def respond(self, question: str, chat_history: List[Message], num_context: int = 2) -> str:
        formatted_history = [
            HumanMessage(content=msg.content) if msg.type == 'human' 
            else AIMessage(content=msg.content)
            for msg in chat_history[-num_context*2:]
        ]
        print(chat_history)
        
        return await self.chain.ainvoke({
            "question": question,
            "chat_history": formatted_history
        })
    

rag_responder_prompt = '''Dựa vào thông tin được cung cấp, hãy trả lời câu hỏi của người dùng
Nếu không có đủ thông tin, hãy trả lời rằng bạn không biết
Thông tin cung cấp:
{context}

Câu hỏi: {question}
'''

class RAGResponder(BaseResponder):
    def __init__(self, llm: ChatOpenAI):
        prompt = ChatPromptTemplate.from_messages([
            ("system", 'Bạn là trợ lý ảo của trường Đại học Bách khoa Hà Nội'),
            ("human", rag_responder_prompt)
        ])
        
        self.chain = prompt | llm | StrOutputParser()

    async def respond(self, question: str, chat_history: List[Message], context: str = "") -> str:
        return await self.chain.ainvoke({
            "question": question,
            "context": context
        })