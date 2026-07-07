"""GRAIL: 4D Human-Object Interaction Generation and Reconstruction."""

from pathlib import Path

# Auto-load .env from project root so users don't need to manually export
# OPENAI_API_KEY / KLING_API_KEY / HF_TOKEN every time.
_project_root = Path(__file__).resolve().parent.parent
_dotenv_path = _project_root / ".env"
try:
    from dotenv import load_dotenv

    load_dotenv(_dotenv_path)
except ImportError:
    # python-dotenv not installed — users must export keys manually.
    # Install it with: pip install python-dotenv
    pass
