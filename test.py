from typing import List, Dict, Optional
from base import BaseRouter, BaseRetriever, BaseReflection, BaseResponder, Message
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage


class LangchainReflection():
    def __init__(self, llm):
        prompt = ChatPromptTemplate.from_messages([
            # ("system", "Bạn là"),
            # MessagesPlaceholder(variable_name="chat_history"),
            ("human", contextualize_q_system_prompt)
        ])
        
        self.chain = prompt | llm | StrOutputParser()

    def expand_query(self, question: str, chat_history, num_context: int = 2) -> str:
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
        res = self.chain.invoke({
            "question": question,
            "chat_history": '\n'.join(chat_history)
        })
        return res

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0
)

contextualize_q_system_prompt = """Cho câu hỏi trước và câu hỏi mới, hãy viết lại câu hỏi để hiểu mà không cần sử dụng câu hỏi trước đó 
Câu hỏi trước: {prev}
Câu hỏi mới: {present}
Lưu ý: Chỉ viết lại nếu THỰC SỰ cần thiết (câu hỏi có liên quan đến ngữ cảnh câu trước đó) , nếu không hãy in lại câu hỏi và không sửa đổi gì thêm
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", contextualize_q_system_prompt),
    ]
)

chain = contextualize_q_prompt | llm | StrOutputParser()

chat_hístory = [
]

print (chain.invoke({
    "prev": "",
    "present": "muốn học kỹ sư thì phải đăng ký như nào"
}))

# a = LangchainReflection(llm)
# print(a.expand_query("", ["human: Muốn học chương trình kỹ sư thì làm như thế nào"], 2))
