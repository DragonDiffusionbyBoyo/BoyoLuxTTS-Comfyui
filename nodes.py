"""
Boyo Lux TTS - ComfyUI Custom Nodes
High-quality voice cloning with LuxTTS
"""

import torch
import numpy as np
import folder_paths
import os
import tempfile
import torchaudio
from zipvoice.luxvoice import LuxTTS


class BoyoLuxTTS_LoadModel:
    """
    Load the LuxTTS model for voice cloning.
    This node should be used once and the MODEL output connected to other nodes.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["cuda", "cpu", "mps"], {"default": "cuda"}),
                "threads": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("LUXTTS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Boyo Lux TTS"
    
    def load_model(self, device, threads):
        """Load the LuxTTS model"""
        try:
            model = LuxTTS('YatharthS/LuxTTS', device=device, threads=threads)
            print(f"✓ LuxTTS model loaded successfully on {device.upper()}")
            return (model,)
        except Exception as e:
            print(f"✗ Error loading LuxTTS model: {str(e)}")
            raise


class BoyoLuxTTS_EncodePrompt:
    """
    Encode a reference audio file for voice cloning.
    The encoded prompt can be reused for multiple text generations.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LUXTTS_MODEL",),
                "audio": ("AUDIO",),
                "duration": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0, "step": 0.5}),
                "rms": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001}),
            }
        }
    
    RETURN_TYPES = ("LUXTTS_PROMPT",)
    RETURN_NAMES = ("encoded_prompt",)
    FUNCTION = "encode_prompt"
    CATEGORY = "Boyo Lux TTS"
    
    def encode_prompt(self, model, audio, duration, rms):
        """Encode the reference audio for voice cloning"""
        try:
            # Extract waveform and sample rate from ComfyUI AUDIO format
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            # Save the audio to the temporary file
            # Ensure waveform is in the right shape for torchaudio (channels, samples)
            if waveform.dim() == 3:  # (batch, samples, channels) or (batch, channels, samples)
                waveform = waveform.squeeze(0)  # Remove batch dimension
            
            if waveform.dim() == 1:  # (samples,) -> (1, samples)
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 2 and waveform.shape[0] > waveform.shape[1]:
                # If (samples, channels), transpose to (channels, samples)
                waveform = waveform.transpose(0, 1)
            
            torchaudio.save(temp_path, waveform.cpu(), sample_rate)
            
            # Encode the prompt using the temporary file
            encoded_prompt = model.encode_prompt(
                temp_path,
                duration=duration,
                rms=rms
            )
            
            # Clean up the temporary file
            try:
                os.unlink(temp_path)
            except:
                pass  # If cleanup fails, it's not critical
            
            print(f"✓ Reference audio encoded successfully")
            return (encoded_prompt,)
            
        except Exception as e:
            print(f"✗ Error encoding prompt: {str(e)}")
            raise


class BoyoLuxTTS_Generate:
    """
    Generate speech from text using the encoded voice prompt.
    This node can be called multiple times with different text using the same model and prompt.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LUXTTS_MODEL",),
                "encoded_prompt": ("LUXTTS_PROMPT",),
                "text": ("STRING", {"default": "", "multiline": True}),
                "num_steps": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "t_shift": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.1}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 5.0, "step": 0.5}),
                "return_smooth": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "Boyo Lux TTS"
    OUTPUT_NODE = True
    
    def generate_speech(self, model, encoded_prompt, text, num_steps, t_shift, speed, guidance_scale, return_smooth):
        """Generate speech from text"""
        try:
            if not text or text.strip() == "":
                raise ValueError("Please provide text to generate")
            
            # Generate speech
            final_wav = model.generate_speech(
                text,
                encoded_prompt,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                speed=speed,
                return_smooth=return_smooth
            )
            
            # Convert to numpy
            audio_np = final_wav.cpu().numpy()
            
            # Ensure proper shape for ComfyUI: (batch, channels, samples)
            # LuxTTS outputs mono audio, so we need (1, 1, samples)
            if audio_np.ndim == 1:
                # (samples,) -> (1, 1, samples)
                audio_np = audio_np[np.newaxis, np.newaxis, :]
            elif audio_np.ndim == 2:
                if audio_np.shape[0] == 1:
                    # (1, samples) -> (1, 1, samples)
                    audio_np = audio_np[:, np.newaxis, :]
                else:
                    # (samples, 1) -> (1, 1, samples)
                    audio_np = audio_np.T[np.newaxis, :, :]
            
            # Convert back to tensor
            audio_tensor = torch.from_numpy(audio_np)
            
            # Return in ComfyUI format: (batch, channels, samples)
            audio_output = {
                "waveform": audio_tensor,
                "sample_rate": 48000
            }
            
            duration = audio_np.shape[2] / 48000
            print(f"✓ Speech generated successfully ({duration:.2f} seconds)")
            print(f"  Output shape: {audio_tensor.shape} (batch, channels, samples)")
            return (audio_output,)
            
        except Exception as e:
            print(f"✗ Error generating speech: {str(e)}")
            raise


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BoyoLuxTTS_LoadModel": BoyoLuxTTS_LoadModel,
    "BoyoLuxTTS_EncodePrompt": BoyoLuxTTS_EncodePrompt,
    "BoyoLuxTTS_Generate": BoyoLuxTTS_Generate,
}

# Display names in ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoLuxTTS_LoadModel": "Lux TTS Load Model",
    "BoyoLuxTTS_EncodePrompt": "Lux TTS Encode Prompt",
    "BoyoLuxTTS_Generate": "Lux TTS Generate",
}
