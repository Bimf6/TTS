#!/usr/bin/env python3
"""
Final Working Fish Speech TTS WebUI with Voice Styles
Includes proper response handling and built-in voice options like 播音 (broadcaster)
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
    GenerateResponse,
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
    "ar": "🇸🇦 Arabic",
    "ru": "🇷🇺 Russian",
    "it": "🇮🇹 Italian",
    "pt": "🇵🇹 Portuguese",
    "nl": "🇳🇱 Dutch",
}

# Voice style configurations with prompt tokens
VOICE_STYLES = {
    "neutral": {
        "name": "🎭 Neutral",
        "prompt": "",
        "description": "Natural, balanced voice"
    },
    "broadcaster": {
        "name": "📻 播音 (Broadcaster)",
        "prompt": "[播音]",
        "description": "Professional news broadcaster style"
    },
    "storyteller": {
        "name": "📚 Storyteller",
        "prompt": "[故事]",
        "description": "Engaging narrative voice"
    },
    "friendly": {
        "name": "😊 Friendly",
        "prompt": "[友善]",
        "description": "Warm, approachable tone"
    },
    "professional": {
        "name": "💼 Professional",
        "prompt": "[专业]",
        "description": "Business, formal tone"
    },
    "energetic": {
        "name": "⚡ Energetic",
        "prompt": "[活力]",
        "description": "Dynamic, enthusiastic"
    },
    "calm": {
        "name": "😌 Calm",
        "prompt": "[平静]",
        "description": "Soothing, relaxed tone"
    },
    "cheerful": {
        "name": "🌟 Cheerful",
        "prompt": "[开朗]",
        "description": "Happy, upbeat voice"
    }
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

def get_voice_style_key(voice_display):
    """Convert display voice to style key"""
    for key, config in VOICE_STYLES.items():
        if config["name"] == voice_display:
            return key
    return "neutral"

def generate_speech_with_voice(
    text, 
    language="🌍 Auto-detect",
    voice_style="🎭 Neutral",
    max_tokens=512,
    temperature=0.7,
    top_p=0.7,
    repetition_penalty=1.5
):
    """Generate speech from text with voice style support"""
    global model_queue, device
    
    if not model_queue:
        return None, "❌ Model not loaded. Please restart the application."
    
    if not text.strip():
        return None, "❌ Please enter some text to generate speech."
    
    try:
        # Get language code and voice style
        lang_code = get_language_code(language)
        voice_key = get_voice_style_key(voice_style)
        voice_config = VOICE_STYLES[voice_key]
        
        # Apply voice prompt if available
        final_text = text
        if voice_config["prompt"]:
            final_text = f"{voice_config['prompt']}{text}"
        
        logger.info(f"Generating speech for: {text[:50]}... (Language: {lang_code}, Voice: {voice_key})")
        if voice_config["prompt"]:
            logger.info(f"Using voice prompt: {voice_config['prompt']}")
        
        logger.info("Starting semantic token generation...")
        
        # Create the generation request with proper parameters
        request_data = {
            "text": final_text,
            "device": device,
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
        
        # Collect all tokens from responses
        all_tokens = []
        
        while True:
            try:
                response: WrappedGenerateResponse = response_queue.get(timeout=60)
                if response.status == "success":
                    if response.response is not None:
                        # Handle the GenerateResponse object properly
                        generate_resp: GenerateResponse = response.response
                        if generate_resp.codes is not None:
                            # Convert tensor to numpy and extend tokens
                            codes_numpy = generate_resp.codes.cpu().numpy()
                            if codes_numpy.ndim == 2:
                                # If 2D, flatten or take first row
                                codes_numpy = codes_numpy.flatten()
                            all_tokens.extend(codes_numpy.tolist())
                        
                        # Check if this is the end signal
                        if generate_resp.action == "next":
                            continue
                        elif generate_resp.action == "sample":
                            # This means generation is complete
                            break
                    else:
                        # No more responses
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
        
        output_file = temp_dir / f"speech_{voice_key}_{uuid.uuid4().hex[:8]}.npy"
        np.save(output_file, semantic_tokens)
        
        logger.info(f"✅ Generated {len(semantic_tokens)} semantic tokens")
        logger.info(f"Saved semantic tokens to: {output_file}")
        
        success_message = f"""✅ Speech generation successful!

📊 Generated: {len(semantic_tokens)} semantic tokens
🌍 Language: {language}
🎭 Voice Style: {voice_style}
📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}
{f'🎤 Voice Prompt: {voice_config["prompt"]}' if voice_config["prompt"] else ''}

📁 File saved: {output_file.name}

🎵 Your speech has been converted to semantic tokens using the {voice_config['description']} style. These tokens represent the AI's understanding of how your text should sound with the selected voice characteristics."""
        
        return str(output_file), success_message
        
    except Exception as e:
        logger.error(f"Speech generation failed: {e}")
        return None, f"❌ Speech generation failed: {str(e)}"

def create_voice_webui():
    """Create the final working WebUI with voice style support"""
    
    with gr.Blocks(
        title="🐠 Fish Speech - Voice Styles",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 2rem; }
        .voice-description { font-size: 0.9em; color: #666; margin-top: 0.5rem; }
        .status-box { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem; }
        """
    ) as app:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🐠 Fish Speech - Voice Styles</h1>
            <p>Professional text-to-speech with multiple voice styles including 播音 (broadcaster)</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Text input
                text_input = gr.Textbox(
                    label="📝 Enter Your Text",
                    placeholder="Type your text here... (支持中文、English、日本語、한국어等多种语言)",
                    lines=5,
                    max_lines=15
                )
                
                with gr.Row():
                    # Language selection
                    language_dropdown = gr.Dropdown(
                        choices=list(LANGUAGES.values()),
                        value="🌍 Auto-detect",
                        label="🌍 Language",
                        scale=1
                    )
                    
                    # Voice style selection
                    voice_style_dropdown = gr.Dropdown(
                        choices=list([config["name"] for config in VOICE_STYLES.values()]),
                        value="🎭 Neutral",
                        label="🎭 Voice Style",
                        scale=1
                    )
                
                # Voice description
                voice_description = gr.HTML(
                    value="<div class='voice-description'>Natural, balanced voice</div>",
                    label=""
                )
                
                # Settings
                with gr.Accordion("⚙️ Advanced Settings", open=False):
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
                    label="📁 Generated Speech Tokens",
                    file_count="single"
                )
                
                status_output = gr.Textbox(
                    label="📊 Generation Status",
                    lines=15,
                    interactive=False
                )
        
        # Voice style examples
        with gr.Row():
            gr.Examples(
                examples=[
                    ["欢迎收听今日新闻，我是主播小鱼。", "🇨🇳 Mandarin Chinese", "📻 播音 (Broadcaster)"],
                    ["Hello everyone, welcome to our broadcast today.", "🇺🇸 English", "📻 播音 (Broadcaster)"],
                    ["从前有一座山，山里有座庙。", "🇨🇳 Mandarin Chinese", "📚 Storyteller"],
                    ["It was a dark and stormy night...", "🇺🇸 English", "📚 Storyteller"],
                    ["大家好，很高兴见到大家！", "🇨🇳 Mandarin Chinese", "😊 Friendly"],
                    ["Good morning, it's a pleasure to meet you all.", "🇺🇸 English", "💼 Professional"],
                ],
                inputs=[text_input, language_dropdown, voice_style_dropdown],
                label="💡 Try These Voice Style Examples"
            )
        
        # Update voice description when style changes
        def update_voice_description(voice_style):
            voice_key = get_voice_style_key(voice_style)
            voice_config = VOICE_STYLES[voice_key]
            prompt_info = f" (Prompt: {voice_config['prompt']})" if voice_config['prompt'] else ""
            return f"<div class='voice-description'>{voice_config['description']}{prompt_info}</div>"
        
        voice_style_dropdown.change(
            update_voice_description,
            inputs=[voice_style_dropdown],
            outputs=[voice_description]
        )
        
        # Generate button click
        generate_btn.click(
            generate_speech_with_voice,
            inputs=[
                text_input,
                language_dropdown,
                voice_style_dropdown,
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
            <p>🐠 <strong>Fish Speech - Voice Styles</strong></p>
            <p>Features: Multi-language support • Professional voice styles • 播音 broadcaster voice • Real-time generation</p>
        </div>
        """)
    
    return app

def main():
    """Main function to initialize and launch the WebUI"""
    global model_queue, device
    
    logger.info("Initializing Fish Speech TTS WebUI with Voice Styles...")
    
    # Initialize model
    model_queue, device = initialize_model()
    
    if not model_queue:
        logger.error("Failed to initialize model. Please check your setup.")
        return
    
    logger.info("Creating Voice Style WebUI...")
    app = create_voice_webui()
    
    logger.info("Starting Voice Style WebUI server...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        debug=False
    )

if __name__ == "__main__":
    main()

