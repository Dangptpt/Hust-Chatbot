import os
from typing import List
from supabase import create_client
from langchain.chat_models import ChatOpenAI
from base import BaseRouter, BaseRetriever, BaseReflection, BaseResponder, Message

class HustChatbot:
    def __init__(
        self,
        router: BaseRouter,
        retriever: BaseRetriever,
        reflection: BaseReflection,
        chitchat_responder: BaseResponder,
        rag_responder: BaseResponder
    ):
        self.router = router
        self.retriever = retriever
        self.reflection = reflection
        self.chitchat_responder = chitchat_responder
        self.rag_responder = rag_responder
        self.chat_history: List[Message] = []

    async def process_message(self, message: str) -> str:
        # Add message to history
        self.chat_history.append(Message(content=message, type='human'))
        
        # Classify question
        query_type = await self.router.classify(message)
        
        if query_type == "chitchat":
            response = await self.chitchat_responder.respond(
                question=message,
                chat_history=self.chat_history
            )
        else:
            # Expand query using reflection
            expanded_query = await self.reflection.expand_query(
                question=message,
                chat_history=self.chat_history
            )
            
            # Retrieve relevant documents
            relevant_docs = await self.retriever.retrieve(expanded_query)
            context = "\n\n".join([doc.get('raw_doc', '') for doc in relevant_docs])
            
            # Generate response
            response = await self.rag_responder.respond(
                question=message,
                chat_history=self.chat_history,
                context=context
            )
        
        # Add response to history
        self.chat_history.append(Message(content=response, type='ai'))
        return response
