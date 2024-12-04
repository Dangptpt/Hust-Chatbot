# file: retrievers.py
from supabase import Client
from typing import List, Dict
from base import BaseRetriever
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from utils.db_services import get_supabase
import os
from dotenv import load_dotenv

load_dotenv()
# model = SentenceTransformer("BAAI/bge-m3")
cache_folder = os.getenv("CACHE_FOLDER")    
model = SentenceTransformer(cache_folder)

class SupabaseRetriever(BaseRetriever):
    def __init__(self):
        self.supabase : Client = get_supabase()

    async def retrieve(self, query: str, match_count=3, match_threshold=0.4) -> List[Dict]:
        # sentence = tokenize(query)
        sentence = query
        embedding = model.encode(sentence).tolist()
        response = self.supabase.rpc(
            'match_documents_v2',
            {
                'query_embedding': embedding, 
                'match_count': match_count,
                'match_threshold': match_threshold
            }
        ).execute()
        return response.data if hasattr(response, 'data') else []
