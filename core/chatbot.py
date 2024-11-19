from typing import List
from base import BaseRouter, BaseRetriever, BaseReflection, BaseResponder, Message
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        self.num_contexts = 2

    async def process_message(self, message: str) -> str:
        # Classify question
        logging.info(f"Question: {message}")
        query_type = await self.router.classify(
            question=message,
            chat_history=self.chat_history,
            num_context=self.num_contexts
        )
        logging.info(f"Agent: {query_type}")

        if query_type == "chitchat":
            response = await self.chitchat_responder.respond(
                question=message,
                chat_history=self.chat_history
            )
            logging.info(f"Answer: {response}")
        else:
            # Expand query using reflection
            expanded_query = await self.reflection.expand_query(
                question=message,
                chat_history=self.chat_history
            )
            logging.info(f"Expanded query: {expanded_query}")
            
            # Retrieve relevant documents
            relevant_docs = await self.retriever.retrieve(expanded_query)
            context = ""
            cnt = 1
            for doc in relevant_docs:
                context += f"Context {cnt}:\n{doc.get('raw_doc', '')}\n\n"
                cnt += 1
            # context = "\n\n".join([doc.get('raw_doc', '') for doc in relevant_docs])
            logging.info(f"retrieval completed")
            # Generate response
            response = await self.rag_responder.respond(
                question=message,
                chat_history=self.chat_history,
                context=context
            )
            logging.info(f"Answer: {response}")

        # Add message to history
        self.chat_history.append(Message(content=message, type='human'))
        self.chat_history.append(Message(content=response, type='ai'))
        return response
