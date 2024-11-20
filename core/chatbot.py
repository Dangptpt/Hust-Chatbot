from typing import List, AsyncGenerator
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
        self.prev_question = ""

    async def stream_process_message(self, message: str) -> AsyncGenerator[str, None]:
        logging.info(f"Question: {message}")
        query_type = await self.router.classify(
            question=message,
            chat_history=self.chat_history,
            num_context=self.num_contexts
        )
        logging.info(f"Agent: {query_type}")

        if query_type == "chitchat":
            async for chunk in self.chitchat_responder.stream_respond(
                question=message,
                chat_history=self.chat_history
            ):
                yield chunk
            response = chunk  # Save final response
        else:
            expanded_query = await self.reflection.expand_query(
                question=message,
                prev_question=self.prev_question
            )
            self.prev_question = expanded_query
            logging.info(f"Expanded query: {expanded_query}")
            
            relevant_docs = await self.retriever.retrieve(expanded_query)
            context = ""
            cnt = 1
            for doc in relevant_docs:
                context += f"Context {cnt}:\n{doc.get('raw_doc', '')}\n\n"
                cnt += 1
            logging.info(f"retrieval {cnt-1} context(s) in total")

            async for chunk in self.rag_responder.stream_respond(
                question=message,
                chat_history=self.chat_history,
                context=context
            ):
                yield chunk
            response = chunk  # Save final response

        self.chat_history.append(Message(content=message, type='human'))
        self.chat_history.append(Message(content=response, type='chatbot'))