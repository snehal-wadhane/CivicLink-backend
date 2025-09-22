import uuid
import os
from supabase import create_client, Client
from dotenv import load_dotenv
import requests
from pathlib import Path
# Load environment variables
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SECRET = os.environ.get("SUPABASE_SECRET")  # make sure spelling matches

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SECRET)

# Bucket name
bucket_name = "product_image"

import uuid
from database import supabase  # your supabase client

# def upload_image_to_supabase(file_path: str):
#     # Read the file
#     with open(file_path, "rb") as f:
#         file_bytes = f.read()

#     # Generate unique filename
#     file_ext = file_path.split(".")[-1]
#     file_name = f"{uuid.uuid4()}.{file_ext}"

#     try:
#         # Upload the file
#         res = supabase.storage.from_(bucket_name).upload(file_name, file_bytes)
        
#         # Check if the upload was successful
#         # The 'res' object itself indicates success or failure via exceptions
#         print(f"File uploaded successfully to path: {res.path}")
#     except Exception as e:
#         # Catch any exceptions raised during the upload process
#         raise Exception(f"Upload failed: {e}")

#     # Get public URL
#     public_url = supabase.storage.from_(bucket_name).get_public_url(file_name)
#     return public_url

def upload_image_to_supabase(file_bytes: bytes, file_ext: str) -> str:
    # Generate unique filename
    file_name = f"{uuid.uuid4()}.{file_ext}"

    try:
        # Upload the file directly from bytes
        supabase.storage.from_(bucket_name).upload(file_name, file_bytes)
        print(f"File uploaded successfully: {file_name}")
    except Exception as e:
        raise Exception(f"Upload failed: {e}")

    # Get public URL
    public_url = supabase.storage.from_(bucket_name).get_public_url(file_name)
    return public_url

import requests

def download_image_from_url(url: str) -> bytes:
    """
    Download an image from a URL and return it as bytes.
    """
    # Clean URL (remove trailing ? if present)
    clean_url = url.split("?")[0]

    response = requests.get(clean_url, headers={
        "User-Agent": "Mozilla/5.0"
    }, timeout=10)

    if response.status_code != 200:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")

    return response.content  # return bytes directly
