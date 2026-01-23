"""
Test Gemini response parsing
"""
import requests
import re

GEMINI_API_KEY = "AIzaSyCLRRUW4uDP4km_aHYBCehNaZpD7dsQnMg"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

def call_gemini_api(prompt, max_tokens=1000):
    """Call Gemini API for text analysis."""
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": max_tokens
        }
    }

    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            print(f"API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None


prompt = """You are a financial analyst. Analyze this sample data for USD/INR:

Recent trends show:
- Economy Tone: -2.5 (negative)
- Conflict Tone: -3.2 (negative)
- Goldstein Scale: -0.15 (slightly negative)
- News Volume: 25% increase

Provide your analysis in EXACTLY this format:

OUTLOOK: [choose one: increase/decrease/stable]
CONFIDENCE: [choose one: low/medium/high]
SENTIMENT_SCORE: [number between -1.0 and 1.0]
VOLATILITY: [choose one: low/medium/high]
REASONING: [2-3 sentences explaining your analysis based on the negative tones and increased news volume]

Use the exact format above."""

print("Testing Gemini with structured prompt...\n")
response = call_gemini_api(prompt, max_tokens=1000)

if response:
    print("=" * 70)
    print("RAW RESPONSE:")
    print("=" * 70)
    print(response)
    print("\n" + "=" * 70)
    print("PARSED RESULTS:")
    print("=" * 70)
    
    # Parse
    result = {
        'outlook': 'stable',
        'confidence': 'medium',
        'sentiment_score': 0.0,
        'volatility': 'medium',
        'reasoning': ''
    }
    
    lines = response.split('\n')
    reasoning_lines = []
    
    for line in lines:
        line_upper = line.upper()
        if 'OUTLOOK:' in line_upper:
            value = line.split(':', 1)[1].strip().lower()
            if 'increase' in value:
                result['outlook'] = 'increase'
            elif 'decrease' in value:
                result['outlook'] = 'decrease'
            else:
                result['outlook'] = 'stable'
        elif 'CONFIDENCE:' in line_upper:
            value = line.split(':', 1)[1].strip().lower()
            if 'high' in value:
                result['confidence'] = 'high'
            elif 'low' in value:
                result['confidence'] = 'low'
        elif 'SENTIMENT_SCORE:' in line_upper or 'SENTIMENT SCORE:' in line_upper:
            try:
                value = line.split(':', 1)[1].strip()
                numbers = re.findall(r'-?\d+\.?\d*', value)
                if numbers:
                    score = float(numbers[0])
                    result['sentiment_score'] = max(-1, min(1, score))
            except:
                pass
        elif 'VOLATILITY:' in line_upper:
            value = line.split(':', 1)[1].strip().lower()
            if 'high' in value:
                result['volatility'] = 'high'
            elif 'low' in value:
                result['volatility'] = 'low'
        elif 'REASONING:' in line_upper:
            reasoning_lines.append(line.split(':', 1)[1].strip())
        elif reasoning_lines:
            reasoning_lines.append(line.strip())
    
    if reasoning_lines:
        result['reasoning'] = ' '.join(reasoning_lines).strip()
    
    if not result['reasoning'] or len(result['reasoning']) < 10:
        result['reasoning'] = f"Based on analysis: {result['outlook']} outlook with {result['confidence']} confidence."
    
    print(f"Outlook: {result['outlook']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Sentiment Score: {result['sentiment_score']}")
    print(f"Volatility: {result['volatility']}")
    print(f"Reasoning: {result['reasoning']}")
else:
    print("Failed to get response")
