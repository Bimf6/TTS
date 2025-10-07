#!/usr/bin/env python3
"""
Simplified Fish Speech WebUI that only does text-to-semantic conversion
Bypasses the decoder model compatibility issues
"""

import os
import gradio as gr
import torch
import numpy as np
from pathlib import Path
from loguru import logger
import pyrootutils

# Setup the root path
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.models.text2semantic.inference import launch_thread_safe_queue

# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"

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
        logger.info("Running on CPU.")
    
    checkpoint_path = "checkpoints/fish-speech-1.5"
    precision = torch.bfloat16
    
    logger.info("Loading Llama model...")
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=checkpoint_path,
        device=device,
        precision=precision,
        compile=False,
    )
    logger.info("Model loaded successfully!")
    
    return llama_queue, device

def generate_semantic_tokens(text, llama_queue, max_new_tokens=512, temperature=0.7, top_p=0.7, repetition_penalty=1.5):
    """Generate semantic tokens from text"""
    try:
        if not text.strip():
            return "Please enter some text!", None, "No text provided"
        
        logger.info(f"Generating semantic tokens for: {text}")
        
        # Create a simple request structure
        class SimpleRequest:
            def __init__(self):
                self.text = text
                self.references = []
                self.reference_id = None
                self.max_new_tokens = max_new_tokens
                self.temperature = temperature
                self.top_p = top_p
                self.repetition_penalty = repetition_penalty
                self.chunk_length = 200
                self.seed = None
                self.use_memory_cache = True
        
        request = SimpleRequest()
        
        # Generate tokens using the queue
        result = llama_queue.put(request)
        
        if hasattr(result, 'codes'):
            codes = result.codes
            logger.info(f"Generated {len(codes)} semantic tokens")
            
            # Save the codes to a temporary file
            output_path = f"temp/webui_codes_{hash(text) % 10000}.npy"
            os.makedirs("temp", exist_ok=True)
            np.save(output_path, codes)
            
            return (
                f"✅ Successfully generated {len(codes)} semantic tokens!",
                output_path,
                f"Semantic tokens saved to: {output_path}\n\nYou can use these tokens with the decoder model when it's available."
            )
        else:
            return "❌ Failed to generate semantic tokens", None, "Generation failed"
            
    except Exception as e:
        logger.error(f"Error generating semantic tokens: {e}")
        return f"❌ Error: {str(e)}", None, f"Error details: {str(e)}"

def create_webui(llama_queue, device):
    """Create the Gradio WebUI"""
    
    with gr.Blocks(title="Fish Speech - Text to Semantic", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🐟 Fish Speech - Text to Semantic Converter
        
        This is a simplified version that converts text to semantic tokens using the Fish Speech model.
        The full audio generation requires compatible decoder models.
        
        **Status**: ✅ Text-to-Semantic model loaded successfully!  
        **Device**: {device}
        """.format(device=device))
        
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text to Convert",
                    placeholder="Enter the text you want to convert to semantic tokens...",
                    lines=3,
                    max_lines=10
                )
                
                with gr.Row():
                    generate_btn = gr.Button("🚀 Generate Semantic Tokens", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                
                with gr.Accordion("Advanced Settings", open=False):
                    max_tokens = gr.Slider(
                        minimum=64,
                        maximum=2048,
                        value=512,
                        step=64,
                        label="Max New Tokens"
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )
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
            
            with gr.Column():
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2
                )
                
                file_output = gr.File(
                    label="Generated Semantic Tokens (.npy file)",
                    interactive=False
                )
                
                details_output = gr.Textbox(
                    label="Details",
                    interactive=False,
                    lines=4
                )
        
        # Event handlers
        def generate_wrapper(text, max_tokens, temp, top_p, rep_penalty):
            return generate_semantic_tokens(text, llama_queue, max_tokens, temp, top_p, rep_penalty)
        
        generate_btn.click(
            fn=generate_wrapper,
            inputs=[text_input, max_tokens, temperature, top_p, repetition_penalty],
            outputs=[status_output, file_output, details_output]
        )
        
        clear_btn.click(
            fn=lambda: ("", "", "", None),
            outputs=[text_input, status_output, details_output, file_output]
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["Hello, this is a test of Fish Speech text-to-speech synthesis."],
                ["The quick brown fox jumps over the lazy dog."],
                ["Fish Speech is an amazing text-to-speech system."],
                ["Welcome to the future of voice synthesis technology."]
            ],
            inputs=[text_input]
        )
        
        gr.Markdown("""
        ### 📝 Instructions:
        1. Enter text in the input box above
        2. Adjust settings if needed (optional)
        3. Click "Generate Semantic Tokens"
        4. Download the generated .npy file containing semantic tokens
        
        ### ℹ️ Note:
        This version only generates semantic tokens. To get actual audio, you would need:
        - Compatible decoder models
        - Full Fish Speech pipeline
        
        The semantic tokens represent the meaning and structure of your text in the model's internal representation.
        """)
    
    return app

def main():
    """Main function to start the WebUI"""
    try:
        logger.info("Initializing Fish Speech WebUI...")
        llama_queue, device = initialize_model()
        
        logger.info("Creating WebUI...")
        app = create_webui(llama_queue, device)
        
        logger.info("Starting WebUI server...")
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_api=True,
            inbrowser=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start WebUI: {e}")
        raise

if __name__ == "__main__":
    main()

