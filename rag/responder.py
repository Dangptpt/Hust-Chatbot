from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from base import BaseRouter, BaseRetriever, BaseResponder, Message
from typing import List
from langchain_core.messages import HumanMessage, AIMessage


class ChitchatResponder(BaseResponder):
    def __init__(self, llm: ChatOpenAI):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là trợ lý ảo của trường Đại học Bách khoa Hà Nội. Hãy trả lời thân thiện bằng tiếng Việt"),
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
        
        return await self.chain.ainvoke({
            "question": question,
            "chat_history": formatted_history
        })

    async def stream_respond(self, question: str, chat_history: List[Message], num_context: int = 2):
        formatted_history = [
            HumanMessage(content=msg.content) if msg.type == 'human' 
            else AIMessage(content=msg.content)
            for msg in chat_history[-num_context*2:]
        ]
        
        async for chunk in self.chain.astream({
            "question": question,
            "chat_history": formatted_history
        }):
            yield chunk
    

rag_responder_prompt = '''Dựa vào thông tin được cung cấp, hãy trả lời câu hỏi của người dùng
Thông tin cung cấp:
{context}

Câu hỏi: {question}
Lưu ý: Nếu bạn không tim thể trả lời từ những thông tin trên, hãy phản hồi "Xin lỗi, tôi chưa có thông tin về {{question}}" và hướng dẫn người dùng tìm kiên thêm thông tin'''

class RAGResponder(BaseResponder):
    def __init__(self, llm: ChatOpenAI):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là trợ lý ảo của trường Đại học Bách khoa Hà Nội. Hãy trả lời thân thiện bằng tiếng Việt"),
            ("human", rag_responder_prompt)
        ])
        
        self.chain = prompt | llm | StrOutputParser()

    async def respond(self, question: str, chat_history: List[Message], context: str = "") -> str:
        return await self.chain.ainvoke({
            "question": question,
            "context": context
        })
    
    async def stream_respond(self, question: str, chat_history: List[Message], context: str = ""):
        async for chunk in self.chain.astream({
            "question": question,
            "context": context
        }):
            yield chunk