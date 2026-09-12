"""
Quick test: Sarvam AI Text-to-Speech (Bulbul v3)

What this does:
1. Sends a short bilingual-friendly sentence to Sarvam's TTS API
2. Gets back audio (as base64 text)
3. Decodes it and saves it as a real .wav file you can play

How to run:
    1. Put your API key below where it says YOUR_API_KEY_HERE
       (or better: set it as an environment variable, see note at bottom)
    2. Run: python test_tts.py
    3. Open the generated output_audio.wav file and listen to it
"""

import base64
import os
from sarvamai import SarvamAI

API_KEY = os.environ.get("SARVAM_API_KEY", "YOUR_API_KEY_HERE")

client = SarvamAI(api_subscription_key=API_KEY)

response = client.text_to_speech.convert(
    text="Namaste! Yeh Sarvam AI ka pehla test hai.",  # "Hello! This is Sarvam AI's first test."
    language_code="hi-IN",
    speaker="kavya",
    model="bulbul:v3",
)

audio_base64 = response.audios[0]
audio_bytes = base64.b64decode(audio_base64)

with open("output_audio.wav", "wb") as f:
    f.write(audio_bytes)

print("Done! Saved audio to output_audio.wav")
print("Request ID:", response.request_id)

# NOTE on keeping your key safe:
# Instead of pasting your key directly into this file, you can instead run:
#   export SARVAM_API_KEY="your_key_here"      (Mac/Linux)
#   set SARVAM_API_KEY=your_key_here           (Windows cmd)
# and then just run: python test_tts.py
# This way the key never sits inside a file you might accidentally share or commit to GitHub.
