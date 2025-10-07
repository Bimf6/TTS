#!/usr/bin/env python3
"""
Simple Working Fish Speech TTS WebUI
Focus on basic functionality that actually works
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

def get_language_code(language_display):
    """Convert display language to code"""
    for code, display in LANGUAGES.items():
        if display == language_display:
            return code
    return "auto"

def generate_speech(
    text, 
    language="🌍 Auto-detect",
    max_tokens=512,
    temperature=0.7,
    top_p=0.7,
    repetition_penalty=1.5
):
    """Generate speech from text using the simplest approach"""
    global model_queue, device
    
    if not model_queue:
        return None, "❌ Model not loaded. Please restart the application."
    
    if not text.strip():
        return None, "❌ Please enter some text to generate speech."
    
    try:
        # Get language code
        lang_code = get_language_code(language)
        logger.info(f"Generating speech for: {text[:50]}... (Language: {lang_code})")
        
        logger.info("Starting semantic token generation...")
        
        # Create the generation request with ONLY supported parameters
        request_data = {
            "text": text,
            "device": device,  # Required parameter
            "max_new_tokens": max_tokens,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "chunk_length": 200,
            "num_samples": 1,
            "iterative_prompt": True,
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
                response: WrappedGenerateResponse = response_queue.get(timeout=60)
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
                return None, "❌ Generation timed out after 60 seconds. Please try again with shorter text."
        
        if not all_tokens:
            return None, "❌ No semantic tokens generated. Please try again."
        
        # Convert tokens to numpy array and save
        semantic_tokens = np.array(all_tokens, dtype=np.int32)
        
        # Create temporary file to save semantic tokens
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        output_file = temp_dir / f"speech_tokens_{uuid.uuid4().hex[:8]}.npy"
        np.save(output_file, semantic_tokens)
        
        logger.info(f"✅ Generated {len(semantic_tokens)} semantic tokens")
        logger.info(f"Saved semantic tokens to: {output_file}")
        
        success_message = f"""✅ Speech generation successful!

📊 Generated: {len(semantic_tokens)} semantic tokens
🌍 Language: {language}
📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}

📁 File saved: {output_file.name}

🎵 Your speech has been converted to semantic tokens which represent the AI's understanding of how your text should sound. These tokens can be processed further to create actual audio files."""
        
        return str(output_file), success_message
        
    except Exception as e:
        logger.error(f"Speech generation failed: {e}")
        return None, f"❌ Speech generation failed: {str(e)}"

def create_simple_webui():
    """Create a simple, working TTS WebUI interface"""
    
    with gr.Blocks(
        title="🐠 Fish Speech - Simple TTS",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 2rem; }
        .status-box { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem; }
        """
    ) as app:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🐠 Fish Speech - Simple TTS</h1>
            <p>Simple and reliable text-to-speech generation</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Text input
                text_input = gr.Textbox(
                    label="📝 Enter Your Text",
                    placeholder="Type your text here... (English, 中文, 日本語, 한국어, etc.)",
                    lines=5,
                    max_lines=15
                )
                
                # Language selection
                language_dropdown = gr.Dropdown(
                    choices=list(LANGUAGES.values()),
                    value="🌍 Auto-detect",
                    label="🌍 Language"
                )
                
                # Simple settings
                with gr.Accordion("⚙️ Settings", open=False):
                    with gr.Row():
                        max_tokens = gr.Slider(
                            minimum=64,
                            maximum=1024,
                            value=512,
                            step=64,
                            label="Max Tokens"
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.5,
                            value=0.7,
                            step=0.1,
                            label="Temperature (Creativity)"
                        )
                    with gr.Row():
                        top_p = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.05,
                            label="Top P (Diversity)"
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
                    label="📁 Generated Speech Tokens",
                    file_count="single"
                )
                
                status_output = gr.Textbox(
                    label="📊 Status",
                    lines=12,
                    interactive=False
                )
        
        # Examples
        with gr.Row():
            gr.Examples(
                examples=[
                    ["Hello world! This is a test of Fish Speech.", "🇺🇸 English"],
                    ["你好世界！这是Fish Speech的测试。", "🇨🇳 Mandarin Chinese"],
                    ["こんにちは世界！これはFish Speechのテストです。", "🇯🇵 Japanese"],
                    ["안녕하세요! Fish Speech 테스트입니다.", "🇰🇷 Korean"],
                    ["Bonjour le monde! Ceci est un test de Fish Speech.", "🇫🇷 French"],
                ],
                inputs=[text_input, language_dropdown],
                label="💡 Try These Examples"
            )
        
        # Generate button click
        generate_btn.click(
            generate_speech,
            inputs=[
                text_input,
                language_dropdown,
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
            <p>🐠 <strong>Fish Speech - Simple TTS</strong></p>
            <p>Converts text to semantic tokens that represent speech patterns</p>
        </div>
        """)
    
    return app

def main():
    """Main function to initialize and launch the WebUI"""
    global model_queue, device
    
    logger.info("Initializing Simple Fish Speech TTS WebUI...")
    
    # Initialize model
    model_queue, device = initialize_model()
    
    if not model_queue:
        logger.error("Failed to initialize model. Please check your setup.")
        return
    
    logger.info("Creating Simple TTS WebUI...")
    app = create_simple_webui()
    
    logger.info("Starting Simple TTS WebUI server...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        debug=False
    )

if __name__ == "__main__":
    main()
