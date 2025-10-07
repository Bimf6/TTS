#!/usr/bin/env python3
"""
Fish Audio TTS - Streamlit Web Application
A beautiful web application for converting text to speech using Fish Audio AI.
"""

import streamlit as st
import requests
import base64
import io
import os
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="Fish Audio TTS",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .reference-section {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border: 2px dashed #27ae60;
        margin: 1rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    .voice-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_URL = "https://api.fish.audio/v1/tts"
MODELS = {
    "Speech 1.5": "speech-1.5",
    "Speech 1.6": "speech-1.6", 
    "Agent X0": "agent-x0",
    "S1": "s1",
    "S1 Mini": "s1-mini"
}

def initialize_session_state():
    """Initialize session state variables"""
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'voices' not in st.session_state:
        st.session_state.voices = []
    if 'selected_voice' not in st.session_state:
        st.session_state.selected_voice = None

def validate_audio_file(uploaded_file) -> tuple[bool, str]:
    """Validate uploaded audio file"""
    if uploaded_file is None:
        return True, ""
    
    valid_types = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/flac', 'audio/ogg']
    max_size = 10 * 1024 * 1024  # 10MB
    
    if uploaded_file.type not in valid_types:
        return False, "Please select a valid audio file (WAV, MP3, FLAC, OGG)"
    
    if uploaded_file.size > max_size:
        return False, "Audio file is too large. Please select a file smaller than 10MB"
    
    return True, ""

def get_voices(api_key: str) -> list:
    """Get available voices from Fish Audio API"""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(f"{API_URL}/voices", headers=headers, timeout=10)
        
        if response.status_code == 200:
            return response.json().get('voices', [])
        else:
            st.error(f"Failed to fetch voices: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching voices: {str(e)}")
        return []

def generate_speech(
    text: str,
    api_key: str,
    model: str = "speech-1.5",
    voice_id: Optional[str] = None,
    reference_audio: Optional[bytes] = None,
    reference_text: Optional[str] = None,
    speech_speed: float = 1.0,
    chunk_length: int = 200
) -> tuple[Optional[bytes], str]:
    """Generate speech using Fish Audio API"""
    
    if not api_key:
        return None, "Please enter your API key"
    
    if not text.strip():
        return None, "Please enter some text to synthesize"
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": text,
            "model": model,
            "chunk_length": chunk_length,
            "speech_speed": speech_speed,
            "format": "wav"
        }
        
        if voice_id:
            payload["voice_id"] = voice_id
        
        if reference_audio and reference_text:
            # Encode reference audio to base64
            reference_audio_b64 = base64.b64encode(reference_audio).decode('utf-8')
            payload["reference_audio"] = reference_audio_b64
            payload["reference_text"] = reference_text
        
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.content, ""
        else:
            error_msg = f"API Error {response.status_code}: {response.text}"
            return None, error_msg
            
    except Exception as e:
        return None, f"Error: {str(e)}"

def main():
    st.markdown('<h1 class="main-header">🐟 Fish Audio TTS</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for API key and settings
    with st.sidebar:
        st.markdown("### 🔑 API Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Fish Audio API Key",
            value=st.session_state.api_key,
            type="password",
            help="Get your API key from fish.audio"
        )
        
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            st.session_state.voices = []
            st.session_state.selected_voice = None
        
        if api_key:
            st.success("✅ API Key configured")
            
            # Load voices
            if st.button("🔄 Refresh Voices") or not st.session_state.voices:
                with st.spinner("Loading voices..."):
                    st.session_state.voices = get_voices(api_key)
            
            if st.session_state.voices:
                st.markdown(f"### 🎤 Available Voices ({len(st.session_state.voices)})")
                
                # Voice selection
                voice_options = {f"{v.get('name', 'Unknown')} ({v.get('language', 'Unknown')})": v['id'] 
                               for v in st.session_state.voices}
                
                selected_voice_name = st.selectbox(
                    "Select Voice",
                    options=list(voice_options.keys()),
                    index=0 if voice_options else None
                )
                
                if selected_voice_name:
                    st.session_state.selected_voice = voice_options[selected_voice_name]
        else:
            st.warning("⚠️ Please enter your API key to get started")
        
        st.markdown("### ⚙️ Settings")
        
        # Model selection
        model = st.selectbox(
            "AI Model",
            options=list(MODELS.keys()),
            index=0,
            help="Choose the Fish Audio model to use"
        )
        
        # Speech speed
        speech_speed = st.slider(
            "Speech Speed",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Adjust the speed of speech generation"
        )
        
        # Chunk length
        chunk_length = st.slider(
            "Chunk Length",
            min_value=100,
            max_value=300,
            value=200,
            step=50,
            help="Smaller chunks may preserve words better"
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">📝 Text Input</h2>', unsafe_allow_html=True)
        
        # Text input
        text_input = st.text_area(
            "Text to Synthesize",
            placeholder="Enter the text you want to convert to speech...",
            height=200,
            help="Enter the text you want to convert to speech"
        )
        
        # Text preprocessing options
        st.markdown("### 🔧 Text Preprocessing")
        
        col_prep1, col_prep2 = st.columns(2)
        
        with col_prep1:
            normalize_text = st.checkbox("Normalize Text", value=True, help="Normalize text for better pronunciation")
            expand_abbreviations = st.checkbox("Expand Abbreviations", value=True, help="Expand common abbreviations")
        
        with col_prep2:
            clean_whitespace = st.checkbox("Clean Whitespace", value=True, help="Remove excessive whitespace")
            add_punctuation = st.checkbox("Add Punctuation", value=False, help="Add punctuation for better flow")
        
        # Processed text preview
        if text_input:
            processed_text = text_input
            
            if clean_whitespace:
                import re
                processed_text = re.sub(r'\s+', ' ', processed_text).strip()
            
            if normalize_text:
                # Basic normalization
                processed_text = processed_text.replace('&', 'and')
                processed_text = processed_text.replace('@', 'at')
            
            if expand_abbreviations:
                abbreviations = {
                    'Dr.': 'Doctor',
                    'Mr.': 'Mister',
                    'Mrs.': 'Misses',
                    'Prof.': 'Professor',
                    'etc.': 'etcetera'
                }
                for abbr, full in abbreviations.items():
                    processed_text = processed_text.replace(abbr, full)
            
            if processed_text != text_input:
                st.markdown("### 📋 Processed Text Preview")
                st.text_area("Preview", value=processed_text, height=100, disabled=True)
                text_input = processed_text
    
    with col2:
        st.markdown('<h2 class="section-header">🎤 Voice Cloning</h2>', unsafe_allow_html=True)
        
        # Reference audio section
        st.markdown('<div class="reference-section">', unsafe_allow_html=True)
        
        # Reference audio upload
        reference_audio = st.file_uploader(
            "Reference Audio (Optional)",
            type=['wav', 'mp3', 'flac', 'ogg'],
            help="Upload a reference audio file for voice cloning"
        )
        
        # Validate reference audio
        if reference_audio:
            is_valid, error_msg = validate_audio_file(reference_audio)
            if is_valid:
                st.success(f"✅ Selected: {reference_audio.name}")
            else:
                st.error(error_msg)
                reference_audio = None
        
        # Reference text input
        reference_text = st.text_area(
            "Reference Text (Optional)",
            placeholder="Enter the text that corresponds to the reference audio...",
            height=100,
            help="Enter the exact text spoken in the reference audio"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate button
        if st.button("🎵 Generate Speech", type="primary", disabled=not text_input.strip() or not api_key):
            if not text_input.strip():
                st.error("Please enter some text to synthesize!")
            elif not api_key:
                st.error("Please enter your API key!")
            else:
                with st.spinner("Generating speech..."):
                    # Prepare reference audio
                    reference_audio_bytes = None
                    if reference_audio:
                        reference_audio_bytes = reference_audio.getvalue()
                    
                    # Generate speech
                    audio_data, error_msg = generate_speech(
                        text=text_input,
                        api_key=api_key,
                        model=MODELS[model],
                        voice_id=st.session_state.selected_voice,
                        reference_audio=reference_audio_bytes,
                        reference_text=reference_text,
                        speech_speed=speech_speed,
                        chunk_length=chunk_length
                    )
                    
                    if audio_data:
                        st.success("🎉 Audio generated successfully!")
                        
                        # Play audio
                        st.audio(audio_data, format='audio/wav')
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Audio",
                            data=audio_data,
                            file_name=f"fish_audio_{hash(text_input) % 10000}.wav",
                            mime="audio/wav"
                        )
                    else:
                        st.error(f"❌ {error_msg}")
    
    # Footer information
    st.markdown("---")
    st.markdown("""
    ### 💡 Tips for Better Results
    
    - **Preventing Missing Words**: Use shorter, clearer sentences
    - **Voice Cloning**: Upload high-quality reference audio (10-30 seconds)
    - **Reference Text**: Enter the exact text spoken in the reference audio
    - **Chunk Length**: Smaller chunks (100-150) may preserve words better
    - **Text Preprocessing**: Keep normalization enabled for better pronunciation
    
    ### 🔗 Resources
    
    - [Fish Audio Website](https://fish.audio)
    - [API Documentation](https://fish.audio/docs)
    - [Voice Gallery](https://fish.audio/voices)
    """)

if __name__ == "__main__":
    main()