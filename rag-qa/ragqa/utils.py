"""Shared utilities for the RAG-QA system."""

import json
import logging
import os
import unicodedata
from pathlib import Path
from typing import Dict, List

import nltk
from transformers import AutoTokenizer


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure logging with timestamps and specified level.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logging("DEBUG")
        >>> logger.info("System initialized")
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure logging format with timestamp
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at {level} level")
    return logger


def load_text_file(filepath: Path) -> str:
    """
    Load text content from .txt or .md files with encoding error handling.
    
    Args:
        filepath: Path to the text file to load
        
    Returns:
        Cleaned text content from the file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a supported text format
        
    Example:
        >>> text = load_text_file(Path("document.txt"))
        >>> print(f"Loaded {len(text)} characters")
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Check file extension
    if filepath.suffix.lower() not in ['.txt', '.md']:
        raise ValueError(f"Unsupported file format: {filepath.suffix}. Only .txt and .md files are supported.")
    
    # Try different encodings
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    text = None
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
    
    if text is None:
        raise ValueError(f"Could not decode file {filepath} with any supported encoding")
    
    # Basic cleaning
    text = text.strip()
    if not text:
        raise ValueError(f"File {filepath} appears to be empty")
    
    return text


def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace, normalizing unicode, and fixing OCR errors.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text string
        
    Example:
        >>> normalized = normalize_text("  Hello   world!  ")
        >>> print(normalized)  # "Hello world!"
    """
    if not text:
        return ""
    
    # Normalize unicode (NFKD decomposition, then recomposition)
    text = unicodedata.normalize('NFKD', text)
    
    # Fix common OCR errors
    ocr_fixes = {
        'ﬁ': 'fi',
        'ﬂ': 'fl',
        'ﬀ': 'ff',
        'ﬃ': 'ffi',
        'ﬄ': 'ffl',
        '–': '-',
        '—': '-',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '…': '...',
    }
    
    for old, new in ocr_fixes.items():
        text = text.replace(old, new)
    
    # Remove extra whitespace
    import re
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def sentence_tokenize(text: str) -> List[str]:
    """
    Tokenize text into sentences using NLTK sentence tokenizer.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of sentence strings
        
    Example:
        >>> sentences = sentence_tokenize("Hello world. How are you?")
        >>> print(sentences)  # ['Hello world.', 'How are you?']
    """
    if not text:
        return []
    
    try:
        # Download punkt tokenizer if not already present
        nltk.download('punkt', quiet=True)
        
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
        
    except Exception as e:
        # Fallback to simple sentence splitting
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences


def save_json(data: Dict, filepath: Path) -> None:
    """
    Save dictionary data to JSON file with pretty formatting.
    
    Args:
        data: Dictionary to save
        filepath: Path where to save the JSON file
        
    Raises:
        TypeError: If data is not JSON serializable
        OSError: If file cannot be written
        
    Example:
        >>> data = {"key": "value", "number": 42}
        >>> save_json(data, Path("output.json"))
    """
    try:
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    except TypeError as e:
        raise TypeError(f"Data is not JSON serializable: {e}")
    except OSError as e:
        raise OSError(f"Cannot write to file {filepath}: {e}")


def load_json(filepath: Path) -> Dict:
    """
    Load JSON data from file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary containing JSON data, or empty dict if file not found
        
    Example:
        >>> data = load_json(Path("config.json"))
        >>> print(data.get("setting", "default"))
    """
    if not filepath.exists():
        return {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        # Return empty dict on any error
        return {}


def compute_token_count(text: str, model_name: str = "bert-base-uncased") -> int:
    """
    Compute token count for text using specified tokenizer model.
    
    Args:
        text: Input text to tokenize
        model_name: Name of the tokenizer model to use
        
    Returns:
        Number of tokens in the text
        
    Example:
        >>> count = compute_token_count("Hello world!", "bert-base-uncased")
        >>> print(f"Token count: {count}")
    """
    if not text:
        return 0
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    except Exception as e:
        # Fallback to simple word count
        return len(text.split())


def ensure_offline_mode() -> None:
    """
    Set environment variables to enable offline mode for transformers and datasets.
    
    This prevents automatic downloads and forces the use of cached models.
    
    Example:
        >>> ensure_offline_mode()
        >>> # Now transformers will work offline
    """
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    
    print("Offline mode enabled:")
    print("  - TRANSFORMERS_OFFLINE=1")
    print("  - HF_DATASETS_OFFLINE=1")
    print("Models will use cached versions only.")
