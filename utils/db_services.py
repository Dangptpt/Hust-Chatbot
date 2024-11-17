import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_KEY")

url: str = SUPABASE_URL
key: str = SUPABASE_SECRET_KEY

def get_supabase():
    supabase: Client = create_client(url, key)
    return supabase