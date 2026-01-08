from transformers import pipeline
from gtts import gTTS
from PIL import Image
import whisper
import os
import re

# Add ffmpeg path
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

print("APP STARTED")

# ---------- Utility ----------
def clean_text(text):
    text = text.encode("ascii", "ignore").decode()  # remove Unicode junk
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------- Load Models ----------
print("Loading Whisper model...")
speech_model = whisper.load_model("base")

print("Loading image-to-text model...")
image_to_text = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base"
)

print("Loading LLM...")
llm = pipeline(
    "text-generation",
    model="gpt2"
)

# ---------- Files ----------
audio_file = "audio.mp3"
image_file = "image.jpg"
user_text = "Explain clearly what is happening"

print("Audio exists:", os.path.exists(audio_file))
print("Image exists:", os.path.exists(image_file))

# ---------- Speech to Text ----------
print("Transcribing audio...")
speech_result = speech_model.transcribe(
    audio_file,
    language="en"   # IMPORTANT FIX
)
speech_text = speech_result["text"]

print("\n🎤 Speech Text:")
print(speech_text)

# ---------- Image to Text ----------
print("\nAnalyzing image...")
image = Image.open(image_file)
image_caption = image_to_text(image)[0]["generated_text"]

print("\n🖼 Image Caption:")
print(image_caption)

# ---------- Combine Prompt ----------
combined_prompt = f"""
Speech Input: {speech_text}
Image Description: {image_caption}
User Text: {user_text}

Give a clear and helpful response.
"""

# ---------- LLM ----------
print("\nGenerating response...")
response = llm(
    combined_prompt,
    max_new_tokens=200,   # FIXED
    do_sample=True
)

final_text = response[0]["generated_text"]
final_text = clean_text(final_text)  # IMPORTANT FIX

print("\n🧠 Final LLM Output:\n")
print(final_text)

# ---------- Text to Speech ----------
print("\nConverting text to speech...")
tts = gTTS(text=final_text, lang="en")
tts.save("output.mp3")

print("\n✅ Audio response saved as output.mp3")
