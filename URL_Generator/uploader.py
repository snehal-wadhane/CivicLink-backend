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
bucket_name = "problem_image"

import uuid
from database import supabase  # your supabase client

def upload_image_to_supabase(file_path: str) -> str:
    """
    Uploads a local file to Supabase Storage and returns the public URL
    """
    # Generate unique filename
    file_ext = file_path.split(".")[-1]
    file_name = f"{uuid.uuid4()}.{file_ext}"

    # Read file bytes
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # Upload file
    res = supabase.storage.from_("problem_image").upload(file_name, file_bytes)

    # Supabase Python client raises exception if upload fails
    # If successful, res.path contains uploaded path
    uploaded_path = res.path  # <-- correct attribute

    # Get public URL
    public_url = supabase.storage.from_("problem_image").get_public_url(uploaded_path)

    return public_url


def download_image_from_url(url: str, save_path: str = None) -> str:
    """
    Download an image from a URL and save it locally.

    Args:
        url (str): The image URL.
        save_path (str, optional): Local path to save the image. 
            If not provided, saves in current directory with original filename.

    Returns:
        str: Path of the saved image.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")

    # Determine filename
    if save_path is None:
        filename = url.split("/")[-1]
        save_path = filename

    # Save the image
    with open(save_path, "wb") as f:
        f.write(response.content)

    return save_path