"""
Test Gemini API connectivity
"""
import requests
import json

GEMINI_API_KEY = "AIzaSyCLRRUW4uDP4km_aHYBCehNaZpD7dsQnMg"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

headers = {
    "Content-Type": "application/json"
}

data = {
    "contents": [{
        "parts": [{
            "text": "Hello, please respond with 'API Working' if you can read this."
        }]
    }],
    "generationConfig": {
        "temperature": 0.3,
        "maxOutputTokens": 100
    }
}

print("Testing Gemini API...")
print(f"URL: {GEMINI_API_URL}")
print(f"API Key: {GEMINI_API_KEY[:20]}...")

try:
    response = requests.post(
        f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
        headers=headers,
        json=data,
        timeout=30
    )
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {response.text[:500]}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ API Working!")
        print(f"Response text: {result['candidates'][0]['content']['parts'][0]['text']}")
    else:
        print(f"\n❌ API Error: {response.status_code}")
        print(f"Error details: {response.text}")
        
except Exception as e:
    print(f"\n❌ Exception: {e}")
