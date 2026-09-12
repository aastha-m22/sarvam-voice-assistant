"""
Quick test: Sarvam AI Translation (Mayura v1)

What this does:
Translates a short English sentence into Hindi, and a short Hindi
sentence into English -- both directions, since Kavya was bilingual.

How to run:
    Same as before: make sure SARVAM_API_KEY is set, then:
    python test_translate.py
"""

import os
from sarvamai import SarvamAI

API_KEY = os.environ.get("SARVAM_API_KEY", "YOUR_API_KEY_HERE")
client = SarvamAI(api_subscription_key=API_KEY)

# English -> Hindi
response1 = client.text.translate(
    input="This item costs three hundred and fifty rupees.",
    source_language_code="en-IN",
    target_language_code="hi-IN",
    model="mayura:v1",
)
print("English -> Hindi:")
print(" ", response1.translated_text)
print()

# Hindi -> English
response2 = client.text.translate(
    input="Yeh product paanch sau rupaye ka hai.",
    source_language_code="hi-IN",
    target_language_code="en-IN",
    model="mayura:v1",
)
print("Hindi -> English:")
print(" ", response2.translated_text)
