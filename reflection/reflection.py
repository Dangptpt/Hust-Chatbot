from typing import List, Dict, Optional
from base import BaseRouter, BaseRetriever, BaseReflection, BaseResponder, Message
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

contextualize_q_system_prompt = """Dựa trên lịch sử câu hỏi và câu hỏi mới nhất của người dùng, hãy viết lại câu hỏi để có thể hiểu mà không cần đến các câu hỏi trước đó.
=== Yêu cầu ===
- KHÔNG trả lời câu hỏi
- Chỉ viết lại câu hỏi nếu cần thiết
- Nếu câu hỏi không liên quan đến những câu hỏi trước đó hãy giữ nguyên, KHÔNG sửa đổi lại câu hỏi
Lịch sử câu hỏi:
{chat_history}
Câu hỏi mới nhất: {question}"""

contextualize_q_system_prompt = """Cho câu hỏi trước và câu hỏi mới, hãy viết lại câu hỏi để hiểu mà không cần sử dụng câu hỏi trước đó 
Câu hỏi trước: {prev_question}
Câu hỏi mới: {question}
Lưu ý: Chỉ viết lại nếu THỰC SỰ cần thiết (câu hỏi có liên quan đến ngữ cảnh câu trước đó) , nếu không hãy in lại câu hỏi và không sửa đổi gì thêm
"""

class LangchainReflection(BaseReflection):
    def __init__(self, llm: ChatOpenAI):
        prompt = ChatPromptTemplate.from_messages([
            # ("system", "Bạn là"),
            # MessagesPlaceholder(variable_name="chat_history"),
            ("human", contextualize_q_system_prompt)
        ])
        
        self.chain = prompt | llm | StrOutputParser()

    async def expand_query(self, question: str, prev_question: str, num_context: int = 2) -> str:
        # formatted_history = [
        #     HumanMessage(content=msg.content) if msg.type == 'human'
        #     else AIMessage(content=msg.content)
        #     for msg in chat_history[-num_context*2:]
        # ]
        # formatted_history = []
        # for msg in chat_history[-num_context*2:]:
        #     if msg.type == 'human':
        #         formatted_history.append(msg.content)
           
        # print('\n'.join(formatted_history))
        return await self.chain.ainvoke({
            "question": question,
            "prev_question": prev_question
            # "chat_history": '\n'.join(formatted_history)
        })