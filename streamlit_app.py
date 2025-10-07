#!/usr/bin/env python3
"""
Fish Audio TTS - Standalone Streamlit App
A completely standalone web application for Fish Audio TTS API.
No dependencies on fish-speech package - uses only Fish Audio API.
"""

import streamlit as st
import requests
import base64
import re

# Page configuration
st.set_page_config(
    page_title="Fish Audio TTS",
    page_icon="🐟",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .reference-section {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border: 2px dashed #27ae60;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def validate_audio_file(uploaded_file):
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

def preprocess_text(text, normalize=True, clean_whitespace=True, expand_abbreviations=True):
    """Preprocess text for better TTS results"""
    processed_text = text
    
    if clean_whitespace:
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    if normalize:
        processed_text = processed_text.replace('&', 'and').replace('@', 'at')
    
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
    
    return processed_text

def generate_speech(text, api_key, model="speech-1.5", voice_id=None, reference_audio=None, reference_text=None, speech_speed=1.0):
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
            "speech_speed": speech_speed,
            "format": "wav"
        }
        
        if voice_id:
            payload["voice_id"] = voice_id
        
        if reference_audio and reference_text:
            reference_audio_b64 = base64.b64encode(reference_audio).decode('utf-8')
            payload["reference_audio"] = reference_audio_b64
            payload["reference_text"] = reference_text
        
        response = requests.post("https://api.fish.audio/v1/tts", json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.content, ""
        else:
            error_msg = f"API Error {response.status_code}: {response.text}"
            return None, error_msg
            
    except Exception as e:
        return None, f"Error: {str(e)}"

def main():
    st.markdown('<h1 class="main-header">🐟 Fish Audio TTS</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔑 API Configuration")
        api_key = st.text_input("Fish Audio API Key", type="password", help="Get your API key from fish.audio")
        
        if api_key:
            st.markdown('<div class="success-box">✅ API Key configured</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">⚠️ Please enter your API key</div>', unsafe_allow_html=True)
        
        st.markdown("### ⚙️ Settings")
        model = st.selectbox("AI Model", ["speech-1.5", "speech-1.6", "agent-x0", "s1", "s1-mini"])
        speech_speed = st.slider("Speech Speed", 0.5, 2.0, 1.0, 0.1)
        
        st.markdown("### 🎤 Voice Selection")
        voice_id = st.text_input("Voice ID (Optional)", help="Enter a specific voice ID if you have one")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Text Input")
        text_input = st.text_area(
            "Text to Synthesize",
            placeholder="Enter the text you want to convert to speech...",
            height=200
        )
        
        # Text preprocessing
        st.markdown("### 🔧 Text Preprocessing")
        col_prep1, col_prep2 = st.columns(2)
        
        with col_prep1:
            normalize_text = st.checkbox("Normalize Text", value=True)
            clean_whitespace = st.checkbox("Clean Whitespace", value=True)
        
        with col_prep2:
            expand_abbreviations = st.checkbox("Expand Abbreviations", value=True)
        
        # Process text
        if text_input:
            processed_text = preprocess_text(
                text_input, 
                normalize=normalize_text,
                clean_whitespace=clean_whitespace,
                expand_abbreviations=expand_abbreviations
            )
            
            if processed_text != text_input:
                st.markdown("### 📋 Processed Text Preview")
                st.text_area("Preview", value=processed_text, height=100, disabled=True)
                text_input = processed_text
    
    with col2:
        st.markdown("### 🎤 Voice Cloning")
        
        # Reference section
        st.markdown('<div class="reference-section">', unsafe_allow_html=True)
        
        # Reference audio upload
        reference_audio = st.file_uploader(
            "Reference Audio (Optional)",
            type=['wav', 'mp3', 'flac', 'ogg'],
            help="Upload a reference audio file for voice cloning"
        )
        
        if reference_audio:
            is_valid, error_msg = validate_audio_file(reference_audio)
            if is_valid:
                st.markdown(f'<div class="success-box">✅ Selected: {reference_audio.name}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="error-box">❌ {error_msg}</div>', unsafe_allow_html=True)
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
                st.markdown('<div class="error-box">❌ Please enter some text to synthesize!</div>', unsafe_allow_html=True)
            elif not api_key:
                st.markdown('<div class="error-box">❌ Please enter your API key!</div>', unsafe_allow_html=True)
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
                        model=model,
                        voice_id=voice_id if voice_id else None,
                        reference_audio=reference_audio_bytes,
                        reference_text=reference_text,
                        speech_speed=speech_speed
                    )
                    
                    if audio_data:
                        st.markdown('<div class="success-box">🎉 Audio generated successfully!</div>', unsafe_allow_html=True)
                        
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
                        st.markdown(f'<div class="error-box">❌ {error_msg}</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 💡 Tips for Better Results
    
    - **Voice Cloning**: Upload high-quality reference audio (10-30 seconds)
    - **Reference Text**: Enter the exact text spoken in the reference audio
    - **Text Preprocessing**: Keep normalization enabled for better pronunciation
    - **Speech Speed**: Adjust speed from 0.5x (slow) to 2.0x (fast)
    
    ### 🔗 Resources
    
    - [Fish Audio Website](https://fish.audio)
    - [API Documentation](https://fish.audio/docs)
    - [Get API Key](https://fish.audio/api-key)
    """)

if __name__ == "__main__":
    main()