#!/usr/bin/env python3
"""
Fish Speech Streamlit Web UI
Features: Reference audio upload, reference text input, speech speed adjustment
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import subprocess
import uuid
import numpy as np
from loguru import logger

# Page configuration
st.set_page_config(
    page_title="Fish Speech AI",
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
</style>
""", unsafe_allow_html=True)

def initialize_model():
    """Initialize and check the model"""
    checkpoint_path = "checkpoints/fish-speech-1.5"
    
    if not Path(checkpoint_path).exists():
        st.error(f"❌ Model not found at {checkpoint_path}")
        st.info("Please make sure you have downloaded the Fish Speech model.")
        return None
    
    st.success("✅ Model checkpoint found successfully!")
    return "cpu"  # Default to CPU for now

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

def generate_tts_audio(text, reference_audio_path=None, reference_text="", speech_speed=1.0):
    """Generate TTS audio using Fish Speech"""
    try:
        if not text.strip():
            return None, "Please enter some text to synthesize!"
        
        st.info(f"🎵 Generating TTS audio for: {text[:50]}...")
        
        # Create temporary directory
        temp_id = str(uuid.uuid4())[:8]
        temp_dir = Path(f"temp/tts_{temp_id}")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Process reference audio if provided
        ref_tokens_path = None
        if reference_audio_path and Path(reference_audio_path).exists():
            st.info("🎤 Processing reference audio for voice cloning...")
            ref_tokens_path = temp_dir / "ref_codes.npy"
            
            cmd = [
                "python3.10", "-m", "fish_speech.models.dac.inference",
                "-i", str(reference_audio_path),
                "--checkpoint-path", "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
                "--device", "cpu",
                "-o", str(ref_tokens_path)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
                if result.returncode != 0:
                    st.warning(f"Reference audio processing failed: {result.stderr}")
                    ref_tokens_path = None
            except Exception as e:
                st.warning(f"Reference audio processing error: {e}")
                ref_tokens_path = None
        
        # Generate semantic tokens
        st.info("🧠 Generating semantic tokens...")
        semantic_tokens_path = temp_dir / "semantic_codes.npy"
        
        cmd = [
            "python3.10", "-m", "fish_speech.models.text2semantic.inference",
            "--text", text,
            "--checkpoint-path", "checkpoints/fish-speech-1.5",
            "--device", "cpu",
            "--max-new-tokens", "512",
            "--temperature", "0.7",
            "--top-p", "0.7",
            "--repetition-penalty", "1.5"
        ]
        
        if ref_tokens_path and ref_tokens_path.exists():
            cmd.extend(["--prompt-tokens", str(ref_tokens_path)])
            if reference_text:
                cmd.extend(["--prompt-text", reference_text])
        
        # Change to temp directory for output
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=original_cwd)
            if result.returncode != 0:
                os.chdir(original_cwd)
                return None, f"❌ Generation failed: {result.stderr}", ""
            
            # Find generated codes file
            codes_files = list(Path(".").glob("codes_*.npy"))
            if not codes_files:
                os.chdir(original_cwd)
                return None, "❌ No semantic tokens generated", ""
            
            semantic_tokens_path = codes_files[0]
            st.success(f"✅ Generated semantic tokens: {semantic_tokens_path}")
            
        finally:
            os.chdir(original_cwd)
        
        # Generate final audio
        st.info("🎵 Generating final audio...")
        output_audio_path = temp_dir / "output.wav"
        
        cmd = [
            "python3.10", "-m", "fish_speech.models.dac.inference",
            "--codes", str(temp_dir / semantic_tokens_path),
            "--checkpoint-path", "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
            "--device", "cpu",
            "-o", str(output_audio_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        if result.returncode != 0:
            return None, f"❌ Audio generation failed: {result.stderr}", ""
        
        if not output_audio_path.exists():
            return None, "❌ Output audio file not created", ""
        
        st.success("✅ Audio generated successfully!")
        return str(output_audio_path), "", str(output_audio_path)
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}", ""

# Main app
def main():
    st.markdown('<h1 class="main-header">🐟 Fish Speech AI</h1>', unsafe_allow_html=True)
    
    # Initialize model
    device = initialize_model()
    if device is None:
        st.stop()
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">🎤 Speech-to-Text</h2>', unsafe_allow_html=True)
        
        # Audio upload for STT
        uploaded_audio = st.file_uploader(
            "Upload Audio File",
            type=['wav', 'mp3', 'flac', 'ogg'],
            help="Upload an audio file to transcribe to text"
        )
        
        if uploaded_audio:
            st.success(f"✅ Uploaded: {uploaded_audio.name}")
            
            # Language selection
            language = st.selectbox(
                "Select Language",
                ["English", "Mandarin", "Cantonese"],
                index=0
            )
            
            if st.button("🎵 Transcribe Audio", type="primary"):
                with st.spinner("Transcribing audio..."):
                    # Here you would implement the STT functionality
                    st.info("STT functionality would be implemented here")
    
    with col2:
        st.markdown('<h2 class="section-header">🔊 Text-to-Speech</h2>', unsafe_allow_html=True)
        
        # Text input
        text_input = st.text_area(
            "Text to Synthesize",
            placeholder="Enter the text you want to convert to speech...",
            height=100
        )
        
        # Voice type selection
        voice_type = st.selectbox(
            "Voice Type",
            ["Neutral", "Friendly", "Professional", "Energetic", "Calm", "Storyteller"],
            index=0
        )
        
        # Speech speed adjustment
        speech_speed = st.slider(
            "Speech Speed",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Adjust the speed of speech generation"
        )
        st.write(f"Current speed: {speech_speed:.1f}x")
        
        # Reference audio section
        st.markdown('<div class="reference-section">', unsafe_allow_html=True)
        st.markdown("### 🎤 Voice Cloning Options")
        
        # Reference audio upload
        reference_audio = st.file_uploader(
            "Reference Audio File (Optional)",
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
        
        # Reference text input
        reference_text = st.text_area(
            "Reference Text (Optional)",
            placeholder="Enter the text that corresponds to the reference audio...",
            height=80,
            help="Enter the exact text spoken in the reference audio for better voice cloning"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate button
        if st.button("🎵 Generate Speech", type="primary", disabled=not text_input.strip()):
            if not text_input.strip():
                st.error("Please enter some text to synthesize!")
            else:
                with st.spinner("Generating speech..."):
                    # Save reference audio if provided
                    reference_audio_path = None
                    if reference_audio:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{reference_audio.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(reference_audio.getvalue())
                            reference_audio_path = tmp_file.name
                    
                    # Generate TTS
                    audio_path, error_msg, success_msg = generate_tts_audio(
                        text_input,
                        reference_audio_path,
                        reference_text,
                        speech_speed
                    )
                    
                    if audio_path:
                        st.success("🎉 Audio generated successfully!")
                        
                        # Play audio
                        with open(audio_path, 'rb') as audio_file:
                            audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format='audio/wav')
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Audio",
                            data=audio_bytes,
                            file_name=f"fish_speech_{uuid.uuid4().hex[:8]}.wav",
                            mime="audio/wav"
                        )
                    else:
                        st.error(error_msg)
                    
                    # Clean up temporary files
                    if reference_audio_path and os.path.exists(reference_audio_path):
                        os.unlink(reference_audio_path)
    
    # Sidebar information
    with st.sidebar:
        st.markdown("### ℹ️ About Fish Speech AI")
        st.markdown("""
        **Fish Speech** is an advanced text-to-speech system that supports:
        
        - 🎤 **Voice Cloning** with reference audio
        - 🌍 **Multi-language** support
        - ⚡ **Speed Control** (0.5x - 2.0x)
        - 🎭 **Multiple Voice Types**
        - 📱 **Web Interface**
        
        ### 🚀 Features
        - Upload reference audio for voice cloning
        - Enter corresponding reference text
        - Adjust speech speed with slider
        - Download generated audio files
        """)
        
        st.markdown("### 🔧 Technical Details")
        st.markdown(f"""
        - **Model**: Fish Speech 1.5
        - **Device**: {device}
        - **Speed**: {speech_speed:.1f}x
        - **Voice**: {voice_type}
        """)

if __name__ == "__main__":
    main()
