import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import CHUNK_OVERLAP, CHUNK_SIZE, RESUMES_DIR
from utils.utils import get_logger, load_pdf

logger = get_logger(__name__)


def _name_from_filename(path: Path) -> str:
    """'Charles_Clark_Resume_47.pdf' -> 'Charles Clark'."""
    stem = re.sub(r"_?Resume_?\d*$", "", path.stem, flags=re.IGNORECASE)
    return stem.replace("_", " ").strip().title()


def _pretty_title(path: Path) -> str:
    return path.stem.replace("_", " ").strip().title()


def load_resumes(directory: Path = RESUMES_DIR) -> List[Document]:
    """One Document per resume PDF (full text). candidate_id = file stem."""
    docs: List[Document] = []
    for pdf in sorted(Path(directory).glob("*.pdf")):
        text = load_pdf(str(pdf))
        if not text.strip():
            logger.warning(f"No text extracted from {pdf.name} - skipped")
            continue
        docs.append(Document(
            page_content=text,
            metadata={"candidate_id": pdf.stem, "name": _name_from_filename(pdf),
                      "source": str(pdf), "role": "resume"},
        ))
    logger.info(f"Loaded {len(docs)} resumes from {directory}")
    return docs


def load_job_descriptions(directory: Path) -> List[Document]:
    docs: List[Document] = []
    for pdf in sorted(Path(directory).glob("*.pdf")):
        text = load_pdf(str(pdf))
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"jd_id": pdf.stem, "title": _pretty_title(pdf),
                          "source": str(pdf), "role": "job_description"},
            ))
    return docs


class DocumentProcessor:
    """OOP wrapper so callers (UI, CLI) have a single ingestion entry point."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def load_resumes(self, directory: Path = RESUMES_DIR) -> List[Document]:
        return load_resumes(directory)

    def load_job_descriptions(self, directory: Path) -> List[Document]:
        return load_job_descriptions(directory)

    def process_pdf(self, pdf_path: Path) -> Document | None:
        text = load_pdf(str(pdf_path))
        if not text.strip():
            return None
        pdf_path = Path(pdf_path)
        return Document(page_content=text,
                        metadata={"candidate_id": pdf_path.stem, "name": _name_from_filename(pdf_path),
                                  "source": str(pdf_path), "role": "resume"})

    def chunk(self, docs: List[Document]) -> List[Document]:
        """Split into overlapping chunks. Each chunk is prefixed with the candidate's name/title
        so a chunk from the middle of a resume still 'knows' whose resume it is."""
        chunks: List[Document] = []
        for doc in docs:
            pieces = self.splitter.split_text(doc.page_content)
            header = f"Candidate: {doc.metadata.get('name', '')}"
            if doc.metadata.get("title"):
                header += f" | {doc.metadata['title']}"
            for i, piece in enumerate(pieces):
                meta = dict(doc.metadata)
                meta["chunk_index"] = i
                chunks.append(Document(page_content=f"{header}\n{piece}", metadata=meta))
        return chunks