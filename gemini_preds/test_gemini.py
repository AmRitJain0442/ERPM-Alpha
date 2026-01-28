"""Quick test to diagnose Gemini API issues."""
import os
import google.generativeai as genai

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Set GEMINI_API_KEY first")
    exit(1)

genai.configure(api_key=api_key)

# Test with simple prompt
model = genai.GenerativeModel('gemini-2.5-flash')

test_prompt = """
Respond with ONLY this JSON, no other text:
```json
{
    "direction": "bullish_usd",
    "adjustment_magnitude": "small",
    "confidence": 7,
    "reasoning": "Test response"
}
```
"""

print("Testing Gemini 2.5 Flash...")
print("-" * 50)

try:
    response = model.generate_content(test_prompt)

    print(f"Response type: {type(response)}")
    print(f"Has .text: {hasattr(response, 'text')}")
    print(f"Has .candidates: {hasattr(response, 'candidates')}")
    print(f"Has .parts: {hasattr(response, 'parts')}")

    # Try .text
    try:
        text = response.text
        print(f"\n.text worked: {text[:200] if text else 'EMPTY'}")
    except Exception as e:
        print(f"\n.text failed: {e}")

    # Check candidates structure
    if hasattr(response, 'candidates') and response.candidates:
        print(f"\nNum candidates: {len(response.candidates)}")
        cand = response.candidates[0]
        print(f"Finish reason: {getattr(cand, 'finish_reason', 'N/A')}")

        if hasattr(cand, 'content') and hasattr(cand.content, 'parts'):
            print(f"Num parts: {len(cand.content.parts)}")
            for i, part in enumerate(cand.content.parts):
                is_thought = hasattr(part, 'thought') and part.thought
                has_text = hasattr(part, 'text') and part.text
                print(f"  Part {i}: thought={is_thought}, has_text={has_text}")
                if has_text:
                    print(f"    Text: {part.text[:100]}...")

    # Check prompt feedback
    if hasattr(response, 'prompt_feedback'):
        print(f"\nPrompt feedback: {response.prompt_feedback}")

except Exception as e:
    print(f"API Error: {e}")
    import traceback
    traceback.print_exc()
