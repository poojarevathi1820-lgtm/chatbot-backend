import os
from dotenv import load_dotenv
from supabase.client import create_client, Client

print("--- Starting Supabase Connection Test ---")

# 1. Load the exact same environment variables as your bot
load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

# 2. Check if the variables were loaded correctly from your .env file
if not supabase_url or not supabase_key:
    print("\n!!!!!!!!!!!!!!!!!! ERROR !!!!!!!!!!!!!!!!!!!")
    print("!!! FAILED TO LOAD .env variables.     !!!")
    print("!!! Make sure your .env file exists and !!!")
    print("!!! contains SUPABASE_URL and SUPABASE_KEY. !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    exit()

print("Successfully loaded URL and Key from .env file.")
print(f"URL: {supabase_url}")

try:
    # 3. Attempt to create a client (connect to your project)
    supabase: Client = create_client(supabase_url, supabase_key)
    print("Successfully created a Supabase client.")

    # 4. Attempt to read from the 'cases' table
    print("Attempting to fetch the first 5 rows from the 'cases' table...")
    
    # This is the actual database query
    response = supabase.table("cases").select("*").limit(5).execute()

    # 5. Analyze the result
    print("\n--- TEST RESULT ---")
    if response.data:
        print("✅ ✅ ✅ SUCCESS! ✅ ✅ ✅")
        print(f"Successfully connected to Supabase and fetched {len(response.data)} rows from the 'cases' table.")
        print("Your .env credentials and RLS policy for reading are CORRECT.")
        print("Sample data fetched:")
        print(response.data)
    else:
        print("🟡🟡🟡 PARTIAL SUCCESS! 🟡🟡🟡")
        print("Successfully connected to Supabase, but the query returned 0 rows.")
        print("This means your credentials and RLS policy are likely CORRECT, but your 'cases' table might be empty.")

except Exception as e:
    # 6. If any step above fails, this will catch the error
    print("\n--- TEST RESULT ---")
    print("❌ ❌ ❌ FAILED! ❌ ❌ ❌")
    print("The test failed to connect to Supabase or fetch data.")
    print("The specific error message is printed below.")
    print("\n---ERROR DETAILS ---")
    print(e)
    print("--------------------------\n")
    
    # Provide the user with the exact solution based on the error
    error_str = str(e).lower()
    if "permission denied for table" in error_str:
        print("SOLUTION: This is a Row Level Security (RLS) error. Your RLS policy is missing or incorrect.")
        print("Go to Authentication -> Policies -> cases and 'Enable read access for everyone'.")
    elif "invalid api key" in error_str or "authentication" in error_str:
        print("SOLUTION: This is an Authentication error. Your SUPABASE_KEY in the .env file is incorrect.")
        print("Go to Project Settings -> API and copy the 'anon' (public) key again.")
    elif "failed to fetch" in error_str or "network" in error_str:
         print("SOLUTION: This is a Network error. Your script could not reach the Supabase URL.")
         print(f"Check your internet connection and make sure the URL is correct: {supabase_url}")
    else:
        print("SOLUTION: An unknown error occurred. Please review the details above.")