"""
Mini Voice Pricing Assistant -- built entirely on Sarvam AI's own APIs

This is a small, single-file demo that mirrors what your Kavya Enterprises
bilingual pricing assistant does, but rebuilt using Sarvam's Text-to-Speech,
Speech-to-Text, and Translation APIs instead of Web Speech API + a local LLM.

THE FLOW (this is the whole point -- read this before running):
  1. We simulate a customer sending a Hindi voice note asking about a price
     (we generate this audio ourselves with TTS, just so this demo doesn't
     need an actual microphone recording)
  2. Speech-to-Text transcribes that voice note into Hindi text
  3. Translation converts the Hindi question into English
  4. A tiny mock catalog (just a Python dictionary -- stand-in for Kavya's
     real 500+ item catalog) looks up a matching answer
  5. Translation converts the English answer back into Hindi
  6. Text-to-Speech speaks the Hindi answer out loud, saved as a .wav file

Run it with: python mini_voice_pricing_assistant.py
"""

import base64
import os
from sarvamai import SarvamAI

API_KEY = os.environ.get("SARVAM_API_KEY", "YOUR_API_KEY_HERE")
client = SarvamAI(api_subscription_key=API_KEY)

# A tiny stand-in for Kavya's real catalog (just for this demo)
MOCK_CATALOG = {
    "rice": "Rice is priced at four hundred rupees per kilogram.",
    "wheat": "Wheat flour is priced at fifty rupees per kilogram.",
    "sugar": "Sugar is priced at forty five rupees per kilogram.",
}


def speak(text, language_code, filename):
    """Text -> speech, saved as a .wav file."""
    response = client.text_to_speech.convert(
        text=text,
        language_code=language_code,
        speaker="kavya",
        model="bulbul:v3",
    )
    audio_bytes = base64.b64decode(response.audios[0])
    with open(filename, "wb") as f:
        f.write(audio_bytes)
    return filename


def listen(filename, language_code):
    """Speech -> text."""
    with open(filename, "rb") as audio_file:
        response = client.speech_to_text.transcribe(
            file=audio_file,
            model="saaras:v3",
            language_code=language_code,
        )
    return response.transcript


def translate(text, source_lang, target_lang):
    response = client.text.translate(
        input=text,
        source_language_code=source_lang,
        target_language_code=target_lang,
        model="mayura:v1",
    )
    return response.translated_text


def find_answer(english_question):
    """Very simple keyword match -- stand-in for Kavya's real catalog lookup."""
    question_lower = english_question.lower()
    for keyword, answer in MOCK_CATALOG.items():
        if keyword in question_lower:
            return answer
    return "Sorry, I could not find that item in the catalog."


# ---- Step 1: simulate the customer's Hindi voice note ----
print("STEP 1: Simulating a customer voice note asking about rice price...")
customer_question_hindi = "Chawal ka daam kya hai?"  # "What is the price of rice?"
speak(customer_question_hindi, "hi-IN", "customer_question.wav")
print("  Created customer_question.wav")
print()

# ---- Step 2: transcribe the voice note ----
print("STEP 2: Transcribing the customer's voice note...")
transcribed_hindi = listen("customer_question.wav", "hi-IN")
print("  Heard:", transcribed_hindi)
print()

# ---- Step 3: translate question to English ----
print("STEP 3: Translating question to English...")
question_english = translate(transcribed_hindi, "hi-IN", "en-IN")
print("  English:", question_english)
print()

# ---- Step 4: look up the answer ----
print("STEP 4: Looking up answer in catalog...")
answer_english = find_answer(question_english)
print("  Answer (English):", answer_english)
print()

# ---- Step 5: translate answer back to Hindi ----
print("STEP 5: Translating answer back to Hindi...")
answer_hindi = translate(answer_english, "en-IN", "hi-IN")
print("  Answer (Hindi):", answer_hindi)
print()

# ---- Step 6: speak the answer ----
print("STEP 6: Speaking the answer out loud...")
speak(answer_hindi, "hi-IN", "assistant_reply.wav")
print("  Created assistant_reply.wav -- play this file to hear the final answer!")
print()

print("Done! Full loop: voice question -> text -> translation -> catalog lookup")
print("-> translation -> voice answer, all using Sarvam AI's own APIs.")
