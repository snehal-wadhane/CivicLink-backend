# db.py
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------
# Test connection
# -------------------------
try:
    # Try fetching 1 row from a table
    response = supabase.table("problems").select("*").limit(1).execute()
    if response.data is not None:
        print("✅ Database connected! Sample data:", response.data)
    else:
        print("✅ Database connected! Table is empty.")
except Exception as e:
    print("❌ Connection failed:", e)
