"""
Boyo Lux TTS - ComfyUI Custom Nodes
High-quality voice cloning with LuxTTS

Based on LuxTTS by YatharthS: https://github.com/ysharma3501/LuxTTS
"""

import os
import sys

# Add the zipvoice directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
zipvoice_path = os.path.join(current_dir, "zipvoice")
if os.path.exists(zipvoice_path) and zipvoice_path not in sys.path:
    sys.path.insert(0, current_dir)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']