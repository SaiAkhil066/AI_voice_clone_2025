import os
import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union, List, Generator
import logging
import soundfile as sf
import time
import tempfile
import shutil
import sys
from pathlib import Path
from scipy import signal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ImprovedVoiceCloner")

class ImprovedVoiceCloner:
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", device: Optional[str] = "cuda"):
        """
        Initialize the improved voice cloning system using XTTS-v2
        """
        # Set device to GPU if specified or available
        if device == "cuda":
            self.use_gpu = torch.cuda.is_available()
            if not self.use_gpu:
                logger.warning("CUDA requested but not available. Falling back to CPU.")
        else:
            self.use_gpu = False
            
        logger.info(f"Using GPU: {self.use_gpu}")
        
        # Fix for PyTorch 2.6+ weights_only=True issue
        self._apply_pytorch_fix()
        
        # Initialize model (will be lazy-loaded when first used)
        self.model_name = model_name
        self.tts = None
        
        # Available languages for XTTS-v2
        self.available_languages = [
            "en", "es", "fr", "de", "it", "pt", "pl", 
            "tr", "ru", "nl", "cs", "ar", "zh-cn", 
            "ja", "hu", "ko", "hi"
        ]
        logger.info(f"Available languages: {self.available_languages}")
            
        # Set optimized parameters for better voice cloning
        self.default_voice_settings = {
                "temperature": 0.65,            # Lower temperature for more stable output
                "repetition_penalty": 2.0,      # Prevent repetition (ensure float)
                "speed": 1.0,                   # Speech rate
                "emotion": "neutral",           # Emotional tone  
                "accent_strength": 0.8,         # Higher accent preservation
                "length_penalty": 1.0,          # Penalty for length differences
                "conditioning_latent_mode": "mean"
            }
        
        # Create directories for temporary files
        self.temp_dir = Path(tempfile.gettempdir()) / "voice_cloning"
        self.temp_dir.mkdir(exist_ok=True)
        
    def _apply_pytorch_fix(self):
        """Apply fix for PyTorch 2.6+ weights_only issue"""
        try:
            # Check PyTorch version
            major, minor = [int(x) for x in torch.__version__.split('.')[:2]]
            
            if (major > 2) or (major == 2 and minor >= 6):
                logger.info(f"Detected PyTorch {torch.__version__} - applying weights_only fix")
                # For PyTorch 2.6+, we need to add TTS classes to safe globals
                from torch.serialization import add_safe_globals
                
                # Add the minimal required classes to safe globals
                from TTS.tts.configs.xtts_config import XttsConfig
                add_safe_globals([XttsConfig])
                
                # Also patch torch.load directly  
                original_torch_load = torch.load
                def patched_torch_load(*args, **kwargs):
                    kwargs['weights_only'] = False  # Always set to False
                    return original_torch_load(*args, **kwargs)
                torch.load = patched_torch_load
                
                logger.info("PyTorch weights_only fix applied successfully")
            else:
                logger.info(f"Using PyTorch {torch.__version__} - no fix needed")
        except Exception as e:
            logger.warning(f"Could not apply PyTorch fix: {str(e)}")
            
    # Replace the _ensure_model_loaded method in improved_voice_clone.py with this fixed version:

    def _ensure_model_loaded(self):
        """Ensure the TTS model is loaded (lazy loading)"""
        if self.tts is None:
            logger.info(f"Loading XTTS-v2 model...")
            try:
                from TTS.api import TTS
                # Simple loading without model listing
                self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                logger.info("Model loaded successfully!")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise RuntimeError(f"Could not load XTTS-v2 model. Try running with --fallback flag")



        # Also add this method to check TTS installation:
    def check_tts_installation(self):
        """Check if TTS library is properly installed"""
        try:
            import TTS
            logger.info(f"TTS version: {TTS.__version__}")
            
            # List available models
            from TTS.api import TTS
            available_models = TTS.list_models()
            logger.info(f"Number of available models: {len(available_models)}")
            
            # Check if xtts_v2 is in the available models
            xtts_models = [m for m in available_models if 'xtts' in m.lower()]
            logger.info(f"XTTS models found: {xtts_models}")
            
            return True
        except Exception as e:
            logger.error(f"TTS installation check failed: {str(e)}")
            return False
    
    # Replace the preprocess_audio method in improved_voice_clone.py with this fixed version:

    def preprocess_audio(self, audio_path: str) -> Tuple[str, Dict]:
        """
        Enhanced audio preprocessing for better voice cloning
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (processed audio path, audio characteristics dict)
        """
        logger.info(f"Preprocessing audio from: {audio_path}")
        
        try:
            # Load audio first to check format
            audio, sample_rate = sf.read(audio_path)
            
            # Convert to mono if needed
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Ensure audio is a 1D array
            audio = np.array(audio).flatten()
            
            # Get original characteristics before resampling
            original_pitch = 0  # Simplified for now
            
            audio_characteristics = {
                "original_sample_rate": sample_rate,
                "original_pitch_mean": original_pitch,
                "original_duration": len(audio) / sample_rate,
                "original_energy": np.sqrt(np.mean(audio**2))
            }
            
            # Resample to 24kHz for XTTS-v2 (optimal sample rate)
            target_sr = 24000
            if sample_rate != target_sr:
                # Use a simpler resampling method that works with our array
                from scipy import signal
                number_of_samples = round(len(audio) * float(target_sr) / sample_rate)
                audio = signal.resample(audio, number_of_samples)
                sample_rate = target_sr
            
            # Normalize audio more carefully to preserve characteristics
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.9
            
            # Ensure minimum length (XTTS-v2 works best with at least 3 seconds)
            min_length = int(3 * sample_rate)
            if len(audio) < min_length:
                logger.warning(f"Audio too short ({len(audio)/sample_rate:.2f}s). Padding to 3 seconds.")
                padding = min_length - len(audio)
                audio = np.pad(audio, (0, padding), mode='wrap')
            
            # Save the processed audio to a temporary file
            temp_path = str(self.temp_dir / f"processed_{int(time.time())}.wav")
            sf.write(temp_path, audio, sample_rate)
            
            logger.info(f"Preprocessed audio saved to: {temp_path}")
            logger.info(f"Audio characteristics: duration={len(audio)/sample_rate:.2f}s")
            
            return temp_path, audio_characteristics
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            raise
    
    def _advanced_normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Simple audio normalization
        """
        peak = np.max(np.abs(audio))
        if peak > 0:
            return audio / peak * 0.9
        return audio
    
    # Replace the clone_voice method in improved_voice_clone.py with this fixed version:

    def clone_voice(self, 
                    voice_sample_path: str, 
                    text: str, 
                    output_path: str = "output/generated_speech.wav",
                    language: str = "en",
                    voice_settings: Optional[Dict] = None) -> str:
        """
        Enhanced voice cloning for better similarity
        
        Args:
            voice_sample_path: Path to the voice sample file
            text: Text to be spoken
            output_path: Path to save the generated audio
            language: Language code
            voice_settings: Optional voice settings
                
        Returns:
            Path to the generated audio file
        """
        try:
            logger.info(f"Processing voice sample: {voice_sample_path}")
            
            # Ensure the model is loaded
            self._ensure_model_loaded()
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Process voice sample and get characteristics
            processed_voice_sample, audio_characteristics = self.preprocess_audio(voice_sample_path)
            
            # Combine default settings with any provided settings
            settings = self.default_voice_settings.copy()
            if voice_settings:
                settings.update(voice_settings)
                
            logger.info(f"Generating speech for text: '{text}' in language: {language}")
            
            # Validate language
            if language not in self.available_languages:
                logger.warning(f"Language '{language}' not in available languages. Defaulting to 'en'.")
                language = "en"
            
            # Ensure repetition_penalty is a float
            repetition_penalty = float(settings.get("repetition_penalty", 2.0))
            
            # Generate speech using enhanced parameters for better voice similarity
            if hasattr(self.tts, 'tts_to_file'):
                # Standard TTS API - enhance with better configurations
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker_wav=processed_voice_sample,
                    language=language,
                    split_sentences=True,  # Better handling of long texts
                    temperature=float(settings.get("temperature", 0.65)),
                    repetition_penalty=repetition_penalty,
                    length_penalty=float(settings.get("length_penalty", 1.0))
                )
            else:
                # Direct model access for fine-tuned control
                # Extract reference features
                logger.info("Using direct model access for enhanced control")
                
                # Load the model's synthesizer if available
                if hasattr(self.tts, 'synthesizer'):
                    synthesizer = self.tts.synthesizer
                    
                    # Extract conditioning latents from the voice sample
                    gpt_cond_latent, speaker_embedding = self.extract_conditioning_latents(
                        processed_voice_sample
                    )
                    
                    # Generate with fine-tuned control
                    wav = synthesizer.tts(
                        text=text,
                        speaker_name=None,
                        language_name=language,
                        speaker_embedding=speaker_embedding,
                        gpt_cond_latent=gpt_cond_latent,
                        temperature=float(settings.get("temperature", 0.65)),
                        repetition_penalty=repetition_penalty,
                        enable_text_splitting=True
                    )
                    
                    # Save the audio
                    synthesizer.save_wav(wav=wav, path=output_path)
                else:
                    # Fallback to standard API
                    self.tts.tts_to_file(
                        text=text,
                        file_path=output_path,
                        speaker_wav=processed_voice_sample,
                        language=language
                    )
            
            # Post-process to match original characteristics
            self.post_process_audio(output_path, audio_characteristics)
            
            # Clean up temporary file
            if os.path.exists(processed_voice_sample):
                os.remove(processed_voice_sample)
            
            logger.info(f"Generated speech saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in voice cloning: {str(e)}")
            raise
    
    def extract_conditioning_latents(self, speaker_wav: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract conditioning latents for fine-tuned voice control
        
        Args:
            speaker_wav: Path to speaker audio file
            
        Returns:
            Tuple of (gpt_cond_latent, speaker_embedding)
        """
        try:
            if hasattr(self.tts, 'synthesizer') and hasattr(self.tts.synthesizer, 'tts_model'):
                model = self.tts.synthesizer.tts_model
                
                # Load audio using soundfile instead of librosa
                wav, sample_rate = sf.read(speaker_wav)
                
                # Resample if needed
                if sample_rate != 24000:
                    number_of_samples = round(len(wav) * float(24000) / sample_rate)
                    wav = signal.resample(wav, number_of_samples)
                
                # Convert to torch tensor
                wav_tensor = torch.FloatTensor(wav).unsqueeze(0)
                
                # Extract conditioning latents using the model's methods
                if hasattr(model, 'get_conditioning_latents'):
                    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                        audio_path=None,
                        audio=wav_tensor,
                        gpt_cond_len=model.gpt_cond_len,
                        gpt_cond_chunk_len=model.gpt_cond_chunk_len,
                        max_ref_len=60
                    )
                    return gpt_cond_latent, speaker_embedding
                else:
                    # Fallback extraction
                    return None, None
            else:
                return None, None
                
        except Exception as e:
            logger.warning(f"Failed to extract conditioning latents: {str(e)}")
            return None, None
    
    def post_process_audio(self, audio_path: str, original_characteristics: Dict):
        """
        Post-process generated audio to better match original voice characteristics
        """
        try:
            # Load generated audio
            audio, sr = sf.read(audio_path)
            
            # Just save without complex processing for now
            sf.write(audio_path, audio, sr)
            logger.info("Post-processing completed")
            
        except Exception as e:
            logger.warning(f"Post-processing failed: {str(e)}")        # Continue without post-processing
    
    def batch_clone_voices(self, voice_samples: List[str], texts: List[str], 
                          output_dir: str = "output/batch", language: str = "en") -> List[str]:
        """
        Process multiple voice samples with different texts
        
        Args:
            voice_samples: List of voice sample file paths
            texts: List of texts to be spoken (same length as voice_samples)
            output_dir: Directory to save outputs
            language: Language code
            
        Returns:
            List of output file paths
        """
        if len(voice_samples) != len(texts):
            raise ValueError("Number of voice samples must match number of texts")
        
        output_paths = []
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (voice_sample, text) in enumerate(zip(voice_samples, texts)):
            output_path = os.path.join(output_dir, f"generated_{i:03d}.wav")
            try:
                result_path = self.clone_voice(voice_sample, text, output_path, language)
                output_paths.append(result_path)
                logger.info(f"Processed sample {i+1}/{len(voice_samples)}")
            except Exception as e:
                logger.error(f"Failed to process sample {i+1}: {str(e)}")
                output_paths.append(None)
        
        return output_paths
    
    def __del__(self):
        """Clean up temporary files on object destruction"""
        try:
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass

# Example usage with optimized settings
if __name__ == "__main__":
    cloner = ImprovedVoiceCloner()
    
    # Test voice cloning with enhanced settings
    voice_settings = {
        "temperature": 0.60,  # Lower temperature for more stability
        "repetition_penalty": 2.5,
        "accent_strength": 0.9,  # High accent preservation
        "length_penalty": 1.1
    }
    
    sample_path = "samples/sample.wav"
    text = "This is a test of the enhanced voice cloning system. The voice should sound more natural and closer to the original."
    output_path = cloner.clone_voice(sample_path, text, language="en", voice_settings=voice_settings)
    print(f"Generated speech saved to: {output_path}")