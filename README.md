# Boyo Lux TTS - ComfyUI Custom Nodes

High-quality voice cloning nodes for ComfyUI using LuxTTS.

## Features

- **SOTA Voice Cloning** - On par with models 10x larger
- **High Quality** - Clear 48kHz speech generation
- **Fast** - 150x realtime on GPU, faster than realtime on CPU
- **Efficient** - Fits within 1GB VRAM
- **Flexible** - Modular node design for complex workflows

## Installation

### Prerequisites

1. 
 Torch >=2.8.0


### Method 1: ComfyUI Manager (Recommended)
ComfyUI Manager will automatically detect and offer to install dependencies from `requirements.txt` when you restart ComfyUI.

### Method 2: Manual Installation

1. Navigate to your ComfyUI custom_nodes directory and clone/copy this repository:
   ```bash
   cd ComfyUI/custom_nodes/
   # Clone or copy BoyoLuxTTS-Comfyui here
   ```

2. **Install dependencies** based on your ComfyUI setup:

   **For Portable ComfyUI (Windows):**
   ```cmd
   # Navigate to ComfyUI root directory
   cd path\to\ComfyUI_windows_portable
   
   # Use the embedded Python
   python_embeded\python.exe -m pip install -r custom_nodes\BoyoLuxTTS-Comfyui\requirements.txt
   ```

   **For Virtual Environment (venv) installations:**
   ```bash
   # Activate your ComfyUI venv first
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   
   # Install requirements
   pip install -r custom_nodes/BoyoLuxTTS-Comfyui/requirements.txt
   ```

   **For Conda environments:**
   ```bash
   # Activate your ComfyUI conda environment
   conda activate comfyui
   
   # Install requirements
   pip install -r custom_nodes/BoyoLuxTTS-Comfyui/requirements.txt
   ```

3. **Restart ComfyUI**

## Nodes

### 1. Lux TTS Load Model
Load the LuxTTS model for voice cloning.

**Inputs:**
- `device`: Device to run on (cuda/cpu/mps)
- `threads`: Number of threads for CPU inference (1-16)

**Outputs:**
- `model`: Loaded LuxTTS model

**Usage:** Load this node once per workflow. The model output can be reused across multiple generations.

---

### 2. Lux TTS Encode Prompt
Encode a reference audio file for voice cloning.

**Inputs:**
- `model`: LuxTTS model from Load Model node
- `audio`: Audio from ComfyUI's Load Audio node (AUDIO type)
- `duration`: Length of reference audio to use (1-10 seconds, default: 5)
- `rms`: Volume control (0.001-0.1, default: 0.01)

**Outputs:**
- `encoded_prompt`: Encoded voice prompt for generation

**Usage:** Connect a Load Audio node to this node's audio input. Encode your reference voice once, then reuse for multiple text generations.

---

### 3. Lux TTS Generate
Generate speech from text using the encoded voice prompt.

**Inputs:**
- `model`: LuxTTS model
- `encoded_prompt`: Encoded voice prompt
- `text`: Text to convert to speech (multiline)
- `num_steps`: Sampling steps (1-16, default: 4) - More steps = better quality but slower
- `t_shift`: Temperature (0.0-1.0, default: 0.9) - Higher = better quality, may have pronunciation errors
- `speed`: Speech speed (0.5-2.0, default: 1.0)
- `guidance_scale`: Text conditioning strength (1.0-5.0, default: 3.0)
- `return_smooth`: Enable smooth mode if you hear metallic sounds

**Outputs:**
- `audio`: Generated audio waveform (48kHz, mono)

**Usage:** Connect to audio preview/save nodes. Can be called multiple times with different text.

## Example Workflow

```
┌─────────────────────────┐
│ Lux TTS Load Model      │
│ device: cuda            │
└───────┬─────────────────┘
        │ model
        ├──────────────────┐
        │                  │
┌───────┴─────────────┐   │
│ Load Audio          │   │
│ audio: ref.wav      │   │
└───────┬─────────────┘   │
        │ audio            │
        │                  │
┌───────▼─────────────┐   │
│ Lux TTS Encode      │   │
│ Prompt              │   │
│ duration: 5         │   │
└───────┬─────────────┘   │
        │ encoded_prompt  │
        │                  │
┌───────▼──────────────────▼┐
│ Lux TTS Generate          │
│ text: "Hello world!"      │
│ num_steps: 4              │
└───────┬───────────────────┘
        │ audio
┌───────▼───────────────────┐
│ Preview Audio / Save      │
└───────────────────────────┘
```

## Tips

- **Reference Audio:** Use at least 3 seconds of clear audio for best voice cloning results
- **Quality vs Speed:** 3-4 sampling steps provide the best efficiency/quality trade-off
- **Pronunciation:** Lower `t_shift` (0.5-0.7) for fewer pronunciation errors but slightly lower quality
- **Metallic Sounds:** Enable `return_smooth` if you hear metallic artifacts (may reduce clarity)
- **Speed Control:** Adjust `speed` parameter to control speaking rate without affecting pitch
- **Batch Processing:** Load model once, encode prompt once, then generate multiple texts efficiently

## System Requirements

- **GPU (CUDA):** Fastest option, requires NVIDIA GPU with CUDA support (~1GB VRAM)
- **MPS:** For Apple Silicon Macs (M1/M2/M3)
- **CPU:** Works on any system but slower (still faster than realtime)

## Credits

### LuxTTS
Created by **YatharthS (Yatharth Sharma)**
- GitHub: https://github.com/ysharma3501/LuxTTS
- Hugging Face: https://huggingface.co/YatharthS/LuxTTS
- Email: yatharthsharma350@gmail.com

LuxTTS is a lightweight ZipVoice-based text-to-speech model designed for high-quality voice cloning and realistic generation at speeds exceeding 150x realtime.

### ZipVoice
LuxTTS is based on the excellent [ZipVoice](https://github.com/k2-fsa/ZipVoice) architecture by k2-fsa.

### Vocos
Uses the [Vocos](https://github.com/gemelo-ai/vocos) vocoder for high-quality 48kHz audio synthesis.

### ComfyUI Integration
Custom node implementation by **Boyo**

## License

This custom node pack is licensed under Apache-2.0, matching the original LuxTTS license.

The underlying LuxTTS model and code are also licensed under Apache-2.0. See the [original repository](https://github.com/ysharma3501/LuxTTS) for details.

## Support

For issues related to:
- **These ComfyUI nodes:** Open an issue on this repository
- **LuxTTS model/core functionality:** Visit the [LuxTTS repository](https://github.com/ysharma3501/LuxTTS)

## Changelog

### v1.0.0
- Initial release
- Three modular nodes: Load Model, Encode Prompt, Generate
- Support for CUDA, CPU, and MPS devices
- 48kHz high-quality audio output in ComfyUI format (batch, channels, samples)
- Native integration with ComfyUI's Load Audio node (no file path input needed)
- Full parameter control (steps, temperature, speed, guidance)
- Tested and validated with NumPy 1.26.4+ and PyTorch 2.x
- Requires zipvoice folder from LuxTTS repository
