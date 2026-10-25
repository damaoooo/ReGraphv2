"""Run the HF dataset builder with python -m DataProcess."""
import os
import sys

os.environ.setdefault('USE_TORCH', '0')

from .cli import app

if __name__ == "__main__":
    if sys.argv[1:2] == ['select']:
        import typer
        from .select_samples import main
        del sys.argv[1]
        typer.run(main)
    elif sys.argv[1:2] == ['export-qwen']:
        from .export_qwen import main
        del sys.argv[1]
        main()
    else:
        app()
