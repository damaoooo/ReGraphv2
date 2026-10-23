"""Run the HF dataset builder with python -m DataProcess."""
import os

os.environ.setdefault('USE_TORCH', '0')

from .cli import app

if __name__ == "__main__":
    app()
