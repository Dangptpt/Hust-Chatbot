from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Message:
    content: str
    type: str  # 'human' or 'ai'

class BaseRouter(ABC):
    @abstractmethod
    async def classify(self, question: str) -> str:
        """Classify the question type"""
        pass

class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str) -> List[Dict]:
        """Retrieve relevant documents"""
        pass

class BaseReflection(ABC):
    @abstractmethod
    async def expand_query(self, question: str, chat_history: List[Message]) -> str:
        """Expand the query based on chat history"""
        pass

class BaseResponder(ABC):
    @abstractmethod
    async def respond(self, question: str, chat_history: List[Message], **kwargs) -> str:
        """Generate response"""
        pass