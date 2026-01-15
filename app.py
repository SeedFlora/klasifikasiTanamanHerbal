"""
Gradio App for Hugging Face Spaces - Indonesian Herbal Plants Classification
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.app import demo

if __name__ == "__main__":
    demo.launch()
