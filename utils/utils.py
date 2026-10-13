import logging
import re
from typing import List

from utils.config import CHUNK_OVERLAP, CHUNK_SIZE


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """Collapse whitespace and drop odd control characters (keeps bullets, $, %, etc.)."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[^\w\s.,;:!?@#$%&*()\-+/='\"|•\[\]<>]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_text(text: str, max_length: int = 8000) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def load_pdf(file_path: str) -> str:
    """Extract all text from a PDF. Returns '' (and logs) on failure instead of crashing."""
    try:
        from langchain_community.document_loaders import PyPDFLoader

        pages = PyPDFLoader(str(file_path)).load()
        text = "\n".join(p.page_content for p in pages)
        # pypdf renders bullet glyphs as control characters (e.g. \x7f); normalise them
        return re.sub(r"[\x7f\uf0b7\u25aa\u25cf\u2022]", "•", text)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to load PDF {file_path}: {e}")
        return ""


def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def is_quota_error(error: Exception) -> bool:
    msg = str(error).lower()
    return "429" in msg or "quota" in msg or "rate limit" in msg or "resource_exhausted" in msg


def llm_text(response) -> str:
    """Return plain text from a LangChain chat response (handles string or block-list content)."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)