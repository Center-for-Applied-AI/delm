"""
Quick test script to verify OpenAI API key and Instructor are working
"""

import os
import openai
from dotenv import load_dotenv

def test_openai_api():
    """Test if OpenAI API key is working with a simple completion."""
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OPENAI_API_KEY found in environment variables")
        print("Make sure you have a .env file with: OPENAI_API_KEY=your_key_here")
        return False
    
    try:
        print("🔑 Testing OpenAI API key...")
        
        # Simple test completion using newer client format
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'Hello, API key is working!' and nothing else."}
            ],
            max_tokens=20,
            temperature=0
        )
        
        result = response.choices[0].message.content
        if result:
            print(f"✅ API key is working! Response: {result.strip()}")
        else:
            print("✅ API key is working! (Empty response)")
        return True
        
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False

def test_instructor():
    """Test if Instructor is working with structured extraction."""
    
    try:
        print("\n🧠 Testing Instructor...")
        
        # Import instructor and pydantic
        import instructor
        from pydantic import BaseModel, Field
        
        # Create a simple test schema
        class TestSchema(BaseModel):
            numbers: list[str] = Field(
                default_factory=list,
                description="Extract all numbers from the text"
            )
        
        # Test text
        test_text = "The price is $50 and quantity is 100 units."
        
        # Patch the client
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        patched_client = instructor.patch(client)
        
        # Test structured extraction
        response = patched_client.chat.completions.create(  # type: ignore
            model="gpt-3.5-turbo",
            response_model=TestSchema,
            messages=[
                {"role": "user", "content": f"Extract numbers from: {test_text}"}
            ],
            temperature=0
        )
        
        print(f"✅ Instructor is working! Extracted: {response.numbers}")
        return True
        
    except ImportError as e:
        print(f"❌ Instructor import failed: {e}")
        print("Install with: pip install instructor")
        return False
    except Exception as e:
        print(f"❌ Instructor test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing DELM dependencies...\n")
    
    # Test OpenAI API first
    api_ok = test_openai_api()
    
    # Test Instructor if API is working
    if api_ok:
        instructor_ok = test_instructor()
        
        if api_ok and instructor_ok:
            print("\n🎉 All tests passed! DELM should work correctly.")
        else:
            print("\n⚠️  Some tests failed. Check the errors above.")
    else:
        print("\n❌ OpenAI API test failed. Fix this first before testing Instructor.") 