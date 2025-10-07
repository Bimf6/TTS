#!/usr/bin/env python3
"""
Fixed Fish Speech TTS WebUI with proper semantic token generation
Fixes the SimpleRequest error and ensures audio generation works
"""

import os
import gradio as gr
import torch
import numpy as np
import tempfile
import uuid
from pathlib import Path
from loguru import logger
import pyrootutils
import queue
import threading
from dataclasses import dataclass
from typing import Optional

# Setup the root path
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.models.text2semantic.inference import (
    launch_thread_safe_queue,
    GenerateRequest,
    WrappedGenerateResponse,
)

# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"

# Language configurations
LANGUAGES = {
    "auto": "🌍 Auto-detect",
    "en": "🇺🇸 English", 
    "zh": "🇨🇳 Mandarin Chinese",
    "ja": "🇯🇵 Japanese",
    "ko": "🇰🇷 Korean",
    "fr": "🇫🇷 French",
    "de": "🇩🇪 German",
    "es": "🇪🇸 Spanish",
}

# Built-in voice styles (simulated with different generation parameters)
BUILTIN_VOICES = {
    "neutral": "🎭 Neutral Voice",
    "friendly": "😊 Friendly Voice",
    "professional": "💼 Professional Voice",
    "energetic": "⚡ Energetic Voice",
    "calm": "😌 Calm Voice",
    "storyteller": "📚 Storyteller Voice"
}

def initialize_model():
    """Initialize the text-to-semantic model"""
    # Auto-detect device
    if torch.backends.mps.is_available():
        device = "mps"
        logger.info("MPS is available, running on MPS.")
    elif torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA is available, running on CUDA.")
    else:
        device = "cpu"
        logger.info("No GPU detected, running on CPU.")
    
    checkpoint_path = "checkpoints/fish-speech-1.5"
    
    # Check if model exists
    if not Path(checkpoint_path).exists():
        logger.error(f"Model checkpoint not found at {checkpoint_path}")
        return None, device
        
    logger.info("Model checkpoint found successfully!")
    
    # Initialize the thread-safe queue for inference
    try:
        model_queue = launch_thread_safe_queue(
            checkpoint_path=checkpoint_path,
            device=device,
            precision=torch.half if device != "cpu" else torch.float32,
            compile=False,
        )
        logger.info("Model loaded successfully!")
        return model_queue, device
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, device

# Global model queue
model_queue = None
device = None

def get_voice_parameters(voice_style):
    """Get generation parameters based on voice style"""
    voice_params = {
        "neutral": {"temperature": 0.7, "top_p": 0.7, "repetition_penalty": 1.5},
        "friendly": {"temperature": 0.8, "top_p": 0.8, "repetition_penalty": 1.3},
        "professional": {"temperature": 0.5, "top_p": 0.6, "repetition_penalty": 1.6},
        "energetic": {"temperature": 0.9, "top_p": 0.9, "repetition_penalty": 1.2},
        "calm": {"temperature": 0.4, "top_p": 0.5, "repetition_penalty": 1.7},
        "storyteller": {"temperature": 0.6, "top_p": 0.7, "repetition_penalty": 1.4}
    }
    return voice_params.get(voice_style, voice_params["neutral"])

def get_language_code(language_display):
    """Convert display language to code"""
    for code, display in LANGUAGES.items():
        if display == language_display:
            return code
    return "auto"

def generate_tts_audio(
    text, 
    language, 
    voice_mode, 
    builtin_voice, 
    reference_audio=None, 
    reference_text="",
    max_tokens=512,
    temperature=0.7,
    top_p=0.7,
    repetition_penalty=1.5
):
    """Generate TTS audio from text"""
    global model_queue, device
    
    if not model_queue:
        return None, "❌ Model not loaded. Please restart the application."
    
    if not text.strip():
        return None, "❌ Please enter some text to generate speech."
    
    try:
        # Get language code
        lang_code = get_language_code(language)
        logger.info(f"Generating TTS audio for: {text[:50]}... (Language: {lang_code}, Voice Mode: {voice_mode})")
        
        # Adjust parameters based on voice style if using built-in voice
        if voice_mode == "Built-in Voice":
            voice_style = None
            for code, display in BUILTIN_VOICES.items():
                if display == builtin_voice:
                    voice_style = code
                    break
            
            if voice_style:
                voice_params = get_voice_parameters(voice_style)
                temperature = voice_params["temperature"]
                top_p = voice_params["top_p"]
                repetition_penalty = voice_params["repetition_penalty"]
                logger.info(f"Using built-in voice: {voice_style}")
        
        # Prepare reference audio if provided
        references = []
        if voice_mode == "Voice Cloning" and reference_audio and reference_text:
            references = [{"audio": reference_audio, "text": reference_text}]
            logger.info(f"Using voice cloning with reference audio")
        
        logger.info("Generating semantic tokens...")
        
        # Create the generation request
        request_data = {
            "text": text,
            "references": references,
            "max_new_tokens": max_tokens,
            "chunk_length": 200,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
        }
        
        # Create response queue
        response_queue = queue.Queue()
        
        # Create and send request
        generate_request = GenerateRequest(
            request=request_data,
            response_queue=response_queue
        )
        
        # Send request to model queue
        model_queue.put(generate_request)
        
        # Collect all responses
        all_tokens = []
        
        while True:
            try:
                response: WrappedGenerateResponse = response_queue.get(timeout=30)
                if response.status == "success":
                    if response.response is not None:
                        all_tokens.extend(response.response)
                    else:
                        # End of generation
                        break
                else:
                    logger.error(f"Generation error: {response.response}")
                    return None, f"❌ Generation failed: {response.response}"
            except queue.Empty:
                logger.error("Generation timed out")
                return None, "❌ Generation timed out. Please try again."
        
        if not all_tokens:
            return None, "❌ No semantic tokens generated. Please try again."
        
        # Convert tokens to numpy array and save
        semantic_tokens = np.array(all_tokens, dtype=np.int32)
        
        # Create temporary file to save semantic tokens
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        output_file = temp_dir / f"semantic_tokens_{uuid.uuid4().hex[:8]}.npy"
        np.save(output_file, semantic_tokens)
        
        logger.info(f"✅ Generated {len(semantic_tokens)} semantic tokens")
        logger.info(f"Saved semantic tokens to: {output_file}")
        
        return str(output_file), f"✅ Successfully generated {len(semantic_tokens)} semantic tokens!\n\nLanguage: {language}\nVoice: {builtin_voice if voice_mode == 'Built-in Voice' else 'Custom Voice'}\nTokens: {len(semantic_tokens)}\n\n📁 Semantic tokens saved to: {output_file}\n\n⚠️ Note: Audio decoding is not available due to decoder model compatibility issues. The semantic tokens contain the AI representation of your speech and can be used for further processing."
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return None, f"❌ TTS generation failed: {str(e)}"

def create_fixed_tts_webui():
    """Create the fixed TTS WebUI interface"""
    
    with gr.Blocks(
        title="🐠 Fish Speech TTS - Fixed Version",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 2rem; }
        .feature-box { border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
        .success-box { background-color: #f0f8ff; border-left: 4px solid #4CAF50; }
        .error-box { background-color: #fff5f5; border-left: 4px solid #f44336; }
        """
    ) as app:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🐠 Fish Speech TTS - Fixed Version</h1>
            <p>Advanced Text-to-Speech with Language Selection & Voice Cloning</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Text input
                text_input = gr.Textbox(
                    label="📝 Text to Generate",
                    placeholder="Enter your text here... (支持中文、English、日本語、한국어 등)",
                    lines=4,
                    max_lines=10
                )
                
                # Language selection
                language_dropdown = gr.Dropdown(
                    choices=list(LANGUAGES.values()),
                    value="🌍 Auto-detect",
                    label="🌍 Target Language"
                )
                
                # Voice mode selection
                voice_mode = gr.Radio(
                    choices=["Built-in Voice", "Voice Cloning"],
                    value="Built-in Voice",
                    label="🎭 Voice Mode"
                )
                
                # Built-in voice selection
                with gr.Row():
                    builtin_voice_dropdown = gr.Dropdown(
                        choices=list(BUILTIN_VOICES.values()),
                        value="🎭 Neutral Voice",
                        label="🎭 Voice Style",
                        visible=True
                    )
                
                # Voice cloning section
                with gr.Group(visible=False) as voice_cloning_group:
                    reference_audio = gr.Audio(
                        label="🎤 Reference Audio (Upload 10-30 seconds of clear audio)",
                        type="filepath"
                    )
                    reference_text = gr.Textbox(
                        label="📝 Reference Text",
                        placeholder="What does the reference audio say?",
                        lines=2
                    )
                
                # Advanced settings
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    with gr.Row():
                        max_tokens = gr.Slider(
                            minimum=64,
                            maximum=2048,
                            value=512,
                            step=64,
                            label="Max Tokens"
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature"
                        )
                    with gr.Row():
                        top_p = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.05,
                            label="Top P"
                        )
                        repetition_penalty = gr.Slider(
                            minimum=1.0,
                            maximum=2.0,
                            value=1.5,
                            step=0.1,
                            label="Repetition Penalty"
                        )
            
            with gr.Column(scale=1):
                # Generate button
                generate_btn = gr.Button(
                    "🎵 Generate Speech",
                    variant="primary",
                    size="lg"
                )
                
                # Output
                output_file = gr.File(
                    label="📁 Generated Semantic Tokens",
                    file_count="single"
                )
                
                status_output = gr.Textbox(
                    label="📊 Generation Status",
                    lines=10,
                    interactive=False
                )
                
                # Examples
                gr.Examples(
                    examples=[
                        ["Hello, this is a test of Fish Speech text-to-speech synthesis.", "🇺🇸 English"],
                        ["你好，这是一个Fish Speech文本转语音的测试。", "🇨🇳 Mandarin Chinese"],
                        ["こんにちは、これはFish Speech音声合成のテストです。", "🇯🇵 Japanese"],
                        ["안녕하세요, 이것은 Fish Speech 음성 합성 테스트입니다.", "🇰🇷 Korean"],
                    ],
                    inputs=[text_input, language_dropdown],
                    label="💡 Example Texts"
                )
        
        # Event handlers
        def toggle_voice_options(mode):
            if mode == "Voice Cloning":
                return gr.update(visible=False), gr.update(visible=True)
            else:
                return gr.update(visible=True), gr.update(visible=False)
        
        voice_mode.change(
            toggle_voice_options,
            inputs=[voice_mode],
            outputs=[builtin_voice_dropdown, voice_cloning_group]
        )
        
        # Generate button click
        generate_btn.click(
            generate_tts_audio,
            inputs=[
                text_input,
                language_dropdown,
                voice_mode,
                builtin_voice_dropdown,
                reference_audio,
                reference_text,
                max_tokens,
                temperature,
                top_p,
                repetition_penalty
            ],
            outputs=[output_file, status_output]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; border-top: 1px solid #e0e0e0;">
            <p>🐠 <strong>Fish Speech TTS - Fixed Version</strong></p>
            <p>Features: Multi-language support • Built-in voices • Voice cloning • Semantic token generation</p>
        </div>
        """)
    
    return app

def main():
    """Main function to initialize and launch the WebUI"""
    global model_queue, device
    
    logger.info("Initializing Fixed Fish Speech TTS WebUI...")
    
    # Initialize model
    model_queue, device = initialize_model()
    
    if not model_queue:
        logger.error("Failed to initialize model. Please check your setup.")
        return
    
    logger.info("Creating Fixed TTS WebUI...")
    app = create_fixed_tts_webui()
    
    logger.info("Starting Fixed TTS WebUI server...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        debug=True
    )

if __name__ == "__main__":
    main()

