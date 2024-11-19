from core.chatbot import HustChatbot
from router.router import LangchainRouter
from rag.retrieval import SupabaseRetriever
from rag.responder import RAGResponder
from reflection.reflection import LangchainReflection
from rag.responder import ChitchatResponder
from langchain_openai.chat_models import ChatOpenAI
import asyncio

async def main():
    # Initialize components
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,    
    )
    
    router = LangchainRouter(llm)
    retriever = SupabaseRetriever()
    reflection = LangchainReflection(llm)
    chitchat_responder = ChitchatResponder(llm)
    rag_responder = RAGResponder(llm)
    
    chatbot = HustChatbot(
        router=router,
        retriever=retriever,
        reflection=reflection,
        chitchat_responder=chitchat_responder,
        rag_responder=rag_responder
    )
    
    # Example usage
    questions = [
        "Vé gửi xe máy hết bao nhiêu tiền ?",
        "xe đạp thì sao",
        "chào em yêu, nay thời tiết thế nào",
    ]
    
    for question in questions:
        response = await chatbot.process_message(question)

if __name__ == "__main__":
    asyncio.run(main())
