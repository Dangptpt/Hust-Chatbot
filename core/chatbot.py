from typing import List, AsyncGenerator
from base import BaseRouter, BaseRetriever, BaseReflection, BaseResponder, Message
import logging
import asyncio
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

    async def get_query_type_and_docs(self, expanded_query: str):
        """Run classification and retrieval concurrently and return results"""
        logging.info("Starting parallel classification and retrieval...")
        
        # Create tasks for both operations
        classify_task = asyncio.create_task(
            self.router.classify(
                question=expanded_query,
                chat_history=self.chat_history,
                num_context=self.num_contexts
            )
        )
        
        retrieve_task = asyncio.create_task(
            self.retriever.retrieve(expanded_query)
        )
        
        # Wait for both tasks to complete
        query_type, relevant_docs = await asyncio.gather(classify_task, retrieve_task)
        
        logging.info(f"Agent: {query_type}")
        logging.info(f"Retrieved {len(relevant_docs)} relevant documents")
        
        return query_type, relevant_docs

    async def stream_process_message(self, message: str) -> AsyncGenerator[str, None]:
        logging.info(f"Question: {message}")
        
        # Expand query first
        expanded_query = await self.reflection.expand_query(
            question=message,
            chat_history=self.chat_history,
        )
        logging.info(f"Expanded query: {expanded_query}")

        # Get query type and relevant docs in parallel
        query_type, relevant_docs = await self.get_query_type_and_docs(expanded_query)

        if query_type == "chitchat":
            async for chunk in self.chitchat_responder.stream_respond(
                question=message,
                chat_history=self.chat_history
            ):
                yield chunk
            response = chunk
        else:
            # Format context from retrieved documents
            context = ""
            cnt = 1
            for doc in relevant_docs:
                context += f"Thông tin {cnt}:\n{doc.get('raw_doc', '')}\n\n"
                cnt += 1

            async for chunk in self.rag_responder.stream_respond(
                question=message,
                chat_history=self.chat_history,
                context=context
            ):
                yield chunk
            response = chunk

        self.chat_history.append(Message(content=message, type='human'))
        self.chat_history.append(Message(content=response, type='chatbot'))