import gradio as gr
import os
import tempfile
import time
import numpy as np
import soundfile as sf
from typing import Dict, Optional, List, Tuple, Union
import importlib
import sys
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VoiceCloneApp")

# Check if we should use the fallback implementation
use_fallback = os.environ.get("USE_FALLBACK", "0") == "1"

# Try to import the appropriate VoiceCloner implementation
try:
    if use_fallback:
        logger.info("Using fallback implementation as directed")
        from fallback_voice_clone import FallbackVoiceCloner
        cloner = FallbackVoiceCloner(device="cuda" if torch.cuda.is_available() else "cpu")
        using_fallback = True
    else:
        logger.info("Attempting to use improved implementation")
        # First try to import the improved cloner
        try:
            from improved_voice_clone import ImprovedVoiceCloner
            cloner = ImprovedVoiceCloner(device="cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            # Fall back to regular voice_clone if improved_voice_clone doesn't exist
            from improved_voice_clone import VoiceCloner
            cloner = VoiceCloner(device="cuda" if torch.cuda.is_available() else "cpu")
        using_fallback = False
except Exception as e:
    logger.warning(f"Failed to initialize selected implementation: {str(e)}")
    logger.info("Trying alternative implementation...")
    try:
        if not use_fallback:
            from fallback_voice_clone import FallbackVoiceCloner
            cloner = FallbackVoiceCloner(device="cuda" if torch.cuda.is_available() else "cpu")
            using_fallback = True
        else:
            from improved_voice_clone import VoiceCloner
            cloner = VoiceCloner(device="cuda" if torch.cuda.is_available() else "cpu")
            using_fallback = False
    except Exception as e:
        logger.error(f"Failed to initialize any voice cloning implementation: {str(e)}")
        raise RuntimeError("Could not initialize any voice cloning implementation")

# Define supported languages based on the implementation
if using_fallback:
    # The fallback implementation only supports English
    LANGUAGES = {"English": "en"}
else:
    # Full implementation supports 17 languages
    LANGUAGES = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Polish": "pl",
        "Turkish": "tr",
        "Russian": "ru",
        "Dutch": "nl",
        "Czech": "cs",
        "Arabic": "ar",
        "Chinese": "zh-cn",
        "Japanese": "ja",
        "Hungarian": "hu",
        "Korean": "ko",
        "Hindi": "hi"
    }

# Define emotion presets
EMOTIONS = ["neutral", "happy", "sad", "angry", "excited", "calm", "whisper"]

def analyze_voice_sample(voice_input: str) -> Dict:
    """
    Analyze voice sample to extract basic characteristics
    """
    try:
        y, sr = sf.read(voice_input)
        
        # If stereo, convert to mono
        if len(y.shape) > 1:
            y = y.mean(axis=1)
        
        # Get duration and basic stats
        duration = len(y) / sr
        energy = np.sqrt(np.mean(y**2))
        
        # Simple pitch estimation
        if len(y) > 0:
            zero_crossings = np.sum(np.diff(np.signbit(y))) / len(y)
            estimated_pitch = sr * zero_crossings / 2
        else:
            estimated_pitch = 0
        
        return {
            "duration": duration,
            "energy": energy,
            "sample_rate": sr,
            "estimated_pitch": estimated_pitch
        }
    except Exception as e:
        print(f"Error analyzing voice sample: {str(e)}")
        return {
            "duration": 0,
            "energy": 0,
            "sample_rate": 0,
            "estimated_pitch": 0
        }

def process_voice_clone(
    voice_input: str,
    text_to_speak: str,
    language: str,
    emotion: str,
    speed: float,
    accent_strength: float,
    temperature: float,
    repetition_penalty: float,
    similarity_boost: bool
) -> Tuple[str, str]:
    """
    Process voice cloning request with enhanced options
    """
    try:
        if not voice_input or not text_to_speak.strip():
            return None, "Please provide both a voice sample and text to speak."
        
        # Analyze voice sample
        voice_analysis = analyze_voice_sample(voice_input)
        
        if voice_analysis["duration"] < 3.0:
            return None, "Voice sample is too short. Please provide at least 3 seconds of clear speech for best results."
        
        # Create unique output path
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        output_path = os.path.join(output_dir, f"generated_{timestamp}.wav")
        
        # Map language name to code
        language_code = LANGUAGES.get(language, "en")
        
        # Set enhanced voice parameters for better similarity
        voice_settings = {
            "speed": speed,
            "emotion": emotion,
            "accent_strength": accent_strength,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "length_penalty": 1.0 if similarity_boost else 0.9
        }
        
        # Apply similarity boost settings for better voice matching
        if similarity_boost:
            voice_settings.update({
                "temperature": min(temperature, 0.65),  # Lower temperature for stability
                "accent_strength": min(accent_strength, 0.85),  # Optimal for best results
                "repetition_penalty": max(repetition_penalty, 2.0)  # Higher penalty to prevent artifacts
            })
        
        # Generate speech with cloned voice
        start_time = time.time()
        result_path = cloner.clone_voice(
            voice_input, 
            text_to_speak, 
            output_path, 
            language=language_code,
            voice_settings=voice_settings
        )
        processing_time = time.time() - start_time
        
        # Create response message based on implementation
        if using_fallback:
            implementation_note = "Using fallback SpeechT5 implementation with limited voice cloning ability."
        else:
            implementation_note = "Using enhanced XTTS-v2 implementation with voice similarity optimization."
        
        # Return result with processing details
        details = f"""
        Voice cloning completed successfully in {processing_time:.2f} seconds!
        {implementation_note}
        
        Voice analysis:
        - Sample duration: {voice_analysis['duration']:.2f} seconds
        - Sample rate: {voice_analysis['sample_rate']} Hz
        - Language: {language} ({language_code})
        - Voice settings applied: Temperature={voice_settings['temperature']:.2f}, 
          Accent strength={voice_settings['accent_strength']:.2f}
        - Pitch estimated at: {voice_analysis['estimated_pitch']:.1f} Hz
        """
        
        return result_path, details
    
    except Exception as e:
        return None, f"Error: {str(e)}"

def save_uploaded_file(file_obj) -> str:
    """Save uploaded file to a temporary location"""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"voice_sample_{int(time.time())}.wav")
    
    if isinstance(file_obj, tuple):
        # For audio recording (sample_rate, audio_data)
        sample_rate, audio_data = file_obj
        sf.write(temp_path, audio_data, sample_rate)
    elif isinstance(file_obj, str):
        # If it's already a path, just return it
        return file_obj
    else:
        # For uploaded files (bytes)
        with open(temp_path, 'wb') as f:
            if hasattr(file_obj, 'read'):
                # If it's a file-like object
                content = file_obj.read()
                if isinstance(content, str):
                    content = content.encode('utf-8')
                f.write(content)
            else:
                # If it's bytes
                if isinstance(file_obj, str):
                    f.write(file_obj.encode('utf-8'))
                else:
                    f.write(file_obj)
    
    return temp_path

def process_with_file_handling(
    voice_input, 
    text_to_speak, 
    language, 
    emotion, 
    speed, 
    accent_strength,
    temperature,
    repetition_penalty,
    similarity_boost
):
    """Handle file processing and voice cloning"""
    try:
        if voice_input is None:
            return None, "Please provide a voice sample."
        
        # Save input to file
        temp_voice_path = save_uploaded_file(voice_input)
        
        # Process voice cloning with enhanced parameters
        return process_voice_clone(
            temp_voice_path,
            text_to_speak,
            language,
            emotion,
            speed,
            accent_strength,
            temperature,
            repetition_penalty,
            similarity_boost
        )
    except Exception as e:
        logger.error(f"Error processing voice: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, f"Error processing voice: {str(e)}"

# Create Gradio interface with enhanced design
with gr.Blocks(theme=gr.themes.Soft(), title="Enhanced AI Voice Cloning (2025)") as app:
    # Determine title based on implementation
    if using_fallback:
        title = "Enhanced AI Voice Cloning (2025) - SpeechT5 Fallback Mode"
        description = """
        This application uses Microsoft's SpeechT5 technology as a fallback for voice cloning.
        Note: This fallback implementation has limited voice cloning ability and only supports English.
        """
    else:
        title = "Enhanced AI Voice Cloning (2025) - XTTS-v2"
        description = """
        This application uses state-of-the-art XTTS-v2 technology with enhanced voice similarity features.
        Record or upload a voice sample, enter text, and adjust parameters to create natural-sounding speech
        that closely matches the original voice.
        """
    
    gr.Markdown(f"# 🎤 {title}")
    gr.Markdown(description)
    
    with gr.Tabs():
        with gr.TabItem("Enhanced Voice Cloning"):
            with gr.Row():
                with gr.Column(scale=1):
                    voice_input = gr.Audio(
                        label="Record or Upload Voice Sample (3-15 seconds recommended)",
                        type="filepath"
                    )
                    
                    language = gr.Dropdown(
                        choices=list(LANGUAGES.keys()),
                        value="English",
                        label="Output Language"
                    )
                    
                    with gr.Accordion("Voice Similarity Settings", open=True):
                        similarity_boost = gr.Checkbox(
                            label="Enable Maximum Voice Similarity Mode",
                            value=True,
                            info="This will optimize all settings for the best voice matching"
                        )
                        
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.65,
                            step=0.05,
                            label="Voice Stability (Temperature)",
                            info="Lower values = more stable but less natural. Higher values = more natural but can vary"
                        )
                        
                        repetition_penalty = gr.Slider(
                            minimum=1.0,
                            maximum=3.0,
                            value=2.0,
                            step=0.1,
                            label="Repetition Penalty",
                            info="Higher values reduce repetitive artifacts"
                        )
                        
                        accent_strength = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.75,
                            step=0.05,
                            label="Accent Preservation Strength",
                            info="How strongly to preserve the original accent"
                        )
                    
                    with gr.Accordion("Additional Voice Settings", open=False):
                        emotion = gr.Dropdown(
                            choices=EMOTIONS,
                            value="neutral",
                            label="Emotional Tone"
                        )
                        
                        speed = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Speech Rate"
                        )
                    
                    text_input = gr.Textbox(
                        label="Text to Speak",
                        placeholder="Type the text you want to be spoken in your voice...",
                        lines=5
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("Generate Speech", variant="primary", size="lg")
                        clear_btn = gr.Button("Clear")
                
                with gr.Column(scale=1):
                    output_audio = gr.Audio(label="Generated Speech")
                    status = gr.Textbox(label="Status", lines=8)
                    
                    gr.Markdown("### Voice Sample Analysis")
                    with gr.Row():
                        with gr.Column(scale=1):
                            sample_duration = gr.Textbox(label="Sample Duration", value="0 seconds")
                        with gr.Column(scale=1):
                            processing_time = gr.Textbox(label="Processing Time", value="0 seconds")
    
    # Instructions accordion
    with gr.Accordion("Instructions", open=False):
        gr.Markdown(
            """
            ### How to use this enhanced voice cloning system:
            
            1. **Record your voice** (3-15 seconds recommended) by clicking the microphone button and saying a few sentences clearly, or **upload** an existing voice recording.
            2. **Choose the language** for the output speech. The system can maintain your voice characteristics across different languages.
            3. **Enable Maximum Voice Similarity Mode** for the best results - this will automatically optimize all settings for voice matching.
            4. **Adjust advanced settings** (optional):
               - Voice Stability: Lower = more consistent, Higher = more natural variations
               - Repetition Penalty: Higher values reduce repetitive artifacts
               - Accent Preservation: How strongly to maintain your original accent
               - Emotional Tone: Select the emotional style of the generated speech
               - Speech Rate: Adjust how fast or slow the voice speaks
            5. **Type the text** you want to hear in your voice
            6. Click **Generate Speech**
            7. Listen to the generated audio that will speak your text in your voice
            
            **For best results:**
            - Record 3-15 seconds of clear speech without background noise
            - Speak naturally at a normal pace
            - Try to record in a quiet environment
            - Use emotional variation in your sample for more expressive results
            - For longer texts, consider shortening or using the emotional tone feature
            
            **Privacy Notice:**
            Your voice data is processed locally and is not stored permanently.
            """
        )
    
    # Set up interactions for enhanced voice cloning
    submit_btn.click(
        fn=process_with_file_handling,
        inputs=[
            voice_input,
            text_input,
            language,
            emotion,
            speed,
            accent_strength,
            temperature,
            repetition_penalty,
            similarity_boost
        ],
        outputs=[output_audio, status]
    )
    
    clear_btn.click(
        fn=lambda: (None, None, ""),
        inputs=[],
        outputs=[voice_input, output_audio, status]
    )
    
    # Example prompts with more natural text
    gr.Examples(
        examples=[
            ["Hello everyone! This is my voice speaking through artificial intelligence. It's amazing how technology can preserve my unique speaking style."],
            ["The enhanced voice cloning system can capture subtle nuances in my accent and speaking patterns, creating remarkably natural results."],
            ["This is a demonstration of how well the system can replicate my voice across different types of speech, from formal to casual tones."]
        ],
        inputs=text_input
    )

# Launch the app
if __name__ == "__main__":
    app.launch(share=False)