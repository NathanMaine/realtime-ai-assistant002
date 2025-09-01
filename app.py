import streamlit as st
import pyaudio
import wave
import whisper
import requests
import pyttsx3
import pandas as pd
import numpy as np
import os
import torch
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Global Constants & Initialization ---
API_KEY = os.getenv("XAI_API_KEY")
if not API_KEY:
    st.error("API_KEY not found in .env file. Please set XAI_API_KEY.")
    st.stop()
API_URL = "https://api.x.ai/v1/chat/completions"
MODEL_RATE = 16000
FILENAME = "temp_audio.wav"

# --- Streamlit Session State Initialization ---
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "actions" not in st.session_state:
    st.session_state.actions = pd.DataFrame(columns=["Task", "Assignee", "Due"])

# --- Cache expensive resources ---
@st.cache_resource
def load_whisper_model():
    """Loads the Whisper model and caches it."""
    return whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_tts_engine():
    """Initializes the text-to-speech engine and caches it."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    return engine

whisper_model = load_whisper_model()
tts_engine = load_tts_engine()

# --- Core Functions ---
def record_and_process():
    """Records a 5-second clip and processes it sequentially."""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    DURATION = 5
    
    p = pyaudio.PyAudio()
    
    try:
        # Step 1: Record Audio
        with st.spinner(f"🔴 Recording for {DURATION} seconds... Speak now!"):
            device_index = p.get_default_input_device_info()['index']
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=MODEL_RATE, input=True,
                            input_device_index=device_index, frames_per_buffer=CHUNK)
            
            frames = []
            for _ in range(0, int(MODEL_RATE / CHUNK * DURATION)):
                data = stream.read(CHUNK)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()

            with wave.open(FILENAME, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(MODEL_RATE)
                wf.writeframes(b''.join(frames))

        # Step 2: Process Audio
        with st.spinner("🧠 Transcribing and analyzing..."):
            result = whisper_model.transcribe(FILENAME, fp16=False)
            transcribed_text = result.get("text", "").strip()

            if not transcribed_text:
                st.warning("No speech detected. Please try recording again.")
                return

            st.write(f"**Transcribed:** *{transcribed_text}*")
            
            llm_response_text = query_llm(transcribed_text)
            if llm_response_text:
                update_ui_with_summary(llm_response_text)

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        p.terminate()
        st.rerun()

def query_llm(text):
    """Sends transcription to the LLM and returns the response."""
    prompt = f'You are a meeting assistant. Summarize the following text and extract action items. Format your response as a JSON object with two keys: "summary" and "actions". The "actions" array should contain objects with "task", "assignee", and "due" keys.\n\nText: {text}'
    payload = {"model": "grok-3", "messages": [{"role": "user", "content": prompt}]}
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"API Error: {e}")
        return ""

def update_ui_with_summary(llm_output_str):
    """Parses the LLM response and updates the UI state."""
    try:
        data = json.loads(llm_output_str)
        new_summary = data.get("summary", "")
        actions_list = data.get("actions", [])
        if new_summary:
            st.session_state.summary += f" {new_summary}\n"
            tts_engine.say(new_summary)
            tts_engine.runAndWait()
        if actions_list:
            new_actions_df = pd.DataFrame(actions_list)
            st.session_state.actions = pd.concat([st.session_state.actions, new_actions_df], ignore_index=True).drop_duplicates()
    except Exception as e:
        st.error(f"Failed to parse or display summary: {e}")

# --- Streamlit UI ---
st.title("AI Meeting Assistant")
st.markdown("Click the button, speak for 5 seconds, and get a summary.")

if st.button("Record & Analyze (5 seconds)"):
    record_and_process()

st.text_area("Summary", value=st.session_state.summary, height=200, key="summary_area")
st.dataframe(st.session_state.actions, use_container_width=True)
