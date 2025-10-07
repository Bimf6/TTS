#!/usr/bin/env python3
"""
Fish Audio TTS - Minimal Streamlit App
Ultra-simple web application for Fish Audio TTS with reference upload features.
"""

import streamlit as st
import requests
import base64

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
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🐟 Fish Audio TTS</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔑 API Configuration")
        api_key = st.text_input("Fish Audio API Key", type="password", help="Get your API key from fish.audio")
        
        if api_key:
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ Please enter your API key")
        
        st.markdown("### ⚙️ Settings")
        model = st.selectbox("AI Model", ["Speech 1.5", "Speech 1.6", "Agent X0", "S1", "S1 Mini"])
        speech_speed = st.slider("Speech Speed", 0.5, 2.0, 1.0, 0.1)
    
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
            processed_text = text_input
            
            if clean_whitespace:
                import re
                processed_text = re.sub(r'\s+', ' ', processed_text).strip()
            
            if normalize_text:
                processed_text = processed_text.replace('&', 'and').replace('@', 'at')
            
            if expand_abbreviations:
                abbreviations = {'Dr.': 'Doctor', 'Mr.': 'Mister', 'Mrs.': 'Misses'}
                for abbr, full in abbreviations.items():
                    processed_text = processed_text.replace(abbr, full)
            
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
            st.success(f"✅ Selected: {reference_audio.name}")
        
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
                    try:
                        # Prepare API request
                        headers = {
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        }
                        
                        payload = {
                            "text": text_input,
                            "model": model.lower().replace(" ", "-"),
                            "speech_speed": speech_speed,
                            "format": "wav"
                        }
                        
                        # Add reference audio if provided
                        if reference_audio and reference_text:
                            reference_audio_b64 = base64.b64encode(reference_audio.getvalue()).decode('utf-8')
                            payload["reference_audio"] = reference_audio_b64
                            payload["reference_text"] = reference_text
                        
                        # Make API request
                        response = requests.post("https://api.fish.audio/v1/tts", json=payload, headers=headers, timeout=30)
                        
                        if response.status_code == 200:
                            st.success("🎉 Audio generated successfully!")
                            
                            # Play audio
                            st.audio(response.content, format='audio/wav')
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Audio",
                                data=response.content,
                                file_name=f"fish_audio_{hash(text_input) % 10000}.wav",
                                mime="audio/wav"
                            )
                        else:
                            st.error(f"❌ API Error {response.status_code}: {response.text}")
                            
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 💡 Tips for Better Results
    
    - **Voice Cloning**: Upload high-quality reference audio (10-30 seconds)
    - **Reference Text**: Enter the exact text spoken in the reference audio
    - **Text Preprocessing**: Keep normalization enabled for better pronunciation
    
    ### 🔗 Resources
    
    - [Fish Audio Website](https://fish.audio)
    - [API Documentation](https://fish.audio/docs)
    """)

if __name__ == "__main__":
    main()
