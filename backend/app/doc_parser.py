from io import BytesIO
from fastapi import UploadFile
from pypdf import PdfReader
from pdfminer.high_level import extract_text
import logging

logging.basicConfig(level=logging.INFO)

async def parse_document(upload: UploadFile) -> str:
    ext = upload.filename.split(".")[-1].lower()
    data = await upload.read()

    if ext == "pdf":
        return _parse_pdf(data)
    if ext in {"txt", "text"}:
        return data.decode("utf-8", errors="ignore").strip()

    raise ValueError("Unsupported file type: only PDF and TXT allowed.")

def _parse_pdf(binary: bytes) -> str:
    """Try PyPDF, fall back to pdfminer.six on failure."""
    try:
        reader = PdfReader(BytesIO(binary))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        return text.strip()
    except Exception as e:
        logging.warning("PyPDF failed (%s); falling back to pdfminer.six", e)
        return extract_text(BytesIO(binary)).strip()

