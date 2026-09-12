"""
Quick test: Sarvam AI Speech-to-Text (Saaras v3)

What this does:
1. Takes the output_audio.wav file you already created with test_tts.py
2. Sends it to Sarvam's Speech-to-Text API
3. Prints back the text it heard

This completes the loop: your original text -> turned into speech -> turned
back into text. If the printed text roughly matches what you typed in
test_tts.py, both APIs are working correctly together.

How to run:
    1. Make sure output_audio.wav (from test_tts.py) is in the same folder
    2. Your SARVAM_API_KEY environment variable should still be set from before
       (if not, set it again: $env:SARVAM_API_KEY="your_key_here")
    3. Run: python test_stt.py
"""

import os
from sarvamai import SarvamAI

API_KEY = os.environ.get("SARVAM_API_KEY", "YOUR_API_KEY_HERE")

client = SarvamAI(api_subscription_key=API_KEY)

with open("output_audio.wav", "rb") as audio_file:
    response = client.speech_to_text.transcribe(
        file=audio_file,
        model="saaras:v3",
        language_code="hi-IN",
    )

print("Sarvam heard:", response.transcript)
print("Detected language:", response.language_code)
print("Request ID:", response.request_id)

print()
print("Compare this to what you originally typed in test_tts.py:")
print('  "Namaste! Yeh Sarvam AI ka pehla test hai."')
