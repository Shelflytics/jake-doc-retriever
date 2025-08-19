"""
A simple script to check if environment variables are loaded correctly from the .env file.
"""
import os
from dotenv import load_dotenv

def check_env_vars():
    """
    Loads environment variables and checks for the GOOGLE_API_KEY.
    """
    # Load variables from .env file into the environment
    load_dotenv()

    print("--- Checking Environment Variables ---")

    # 1. Check for GOOGLE_API_KEY
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        # For security, show only the first few and last few characters
        masked_key = f"{google_key[:4]}...{google_key[-4:]}"
        print(f"✅ GOOGLE_API_KEY found: {masked_key}")
    else:
        print("❌ GOOGLE_API_KEY not found.")
        print("   Please make sure you have a .env file with 'GOOGLE_API_KEY=your_key_here'.")

    # 2. Check for other important variables (optional)
    ai_model = os.getenv("AI_MODEL")
    if ai_model:
        print(f"✅ AI_MODEL found: {ai_model}")
    else:
        print("❌ AI_MODEL not found.")

    index_dir = os.getenv("INDEX_DIR")
    if index_dir:
        print(f"✅ INDEX_DIR found: {index_dir}")
    else:
        print("❌ INDEX_DIR not found.")
        
    print("------------------------------------")


if __name__ == "__main__":
    check_env_vars()
