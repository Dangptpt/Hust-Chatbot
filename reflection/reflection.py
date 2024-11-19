from typing import List, Dict, Optional
from base import BaseRouter, BaseRetriever, BaseReflection, BaseResponder, Message
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

contextualize_q_system_prompt = """Dựa trên lịch sử trò chuyện và câu hỏi mới nhất từ người dùng \
(có thể liên quan đến ngữ cảnh trong lịch sử trò chuyện) , hãy viết lại câu hỏi thành \
một câu độc lập có thể hiểu mà không cần đến lịch sử trò chuyện. KHÔNG trả lời câu hỏi, \
chỉ viết lại nếu cần và nếu không thì giữ nguyên."""

class LangchainReflection(BaseReflection):
    def __init__(self, llm: ChatOpenAI):
        prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        self.chain = prompt | llm | StrOutputParser()

    async def expand_query(self, question: str, chat_history: List[Message], num_context: int = 2) -> str:
        formatted_history = [
            HumanMessage(content=msg.content) if msg.type == 'human'
            else AIMessage(content=msg.content)
            for msg in chat_history[-num_context*2:]
        ]
        
        return await self.chain.ainvoke({
            "question": question,
            "chat_history": formatted_history
        })