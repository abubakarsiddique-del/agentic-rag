import asyncio
import io
import os
from pathlib import Path
from typing import Iterable, AsyncIterable

from langchain_core.documents import Document
from pypdf import PdfReader


def _validate_file_size(file_bytes: bytes, file_name: str) -> None:
    maximum_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    if len(file_bytes) > maximum_bytes:
        raise ValueError(
            f"{file_name} exceeds the {maximum_bytes} byte upload size limit. "
            "Please upload a smaller PDF or text file."
        )


def _decode_text_bytes(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            return file_bytes.decode("utf-8", errors="replace")


def load_uploaded_documents(uploaded_files: Iterable) -> list[Document]:
    documents: list[Document] = []

    for uploaded_file in uploaded_files:
        file_name = getattr(uploaded_file, "name", "") or "Unknown file"
        normalized_name = os.path.basename(file_name)
        file_bytes = uploaded_file.getvalue()
        _validate_file_size(file_bytes, normalized_name)

        extension = os.path.splitext(normalized_name)[1].lower()

        if extension == ".pdf":
            reader = PdfReader(io.BytesIO(file_bytes))
            if getattr(reader, "is_encrypted", False):
                raise ValueError("This PDF is encrypted or protected and cannot be read. Please upload an unprotected PDF.")
            page_texts: list[tuple[int, str]] = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text and text.strip():
                    page_texts.append((page_number, text))
            if not page_texts:
                raise ValueError("This PDF appears to be scanned or image-based without readable text. Please upload a text-based PDF or a plain text file.")
            for page_number, text in page_texts:
                documents.append(Document(page_content=text, metadata={"source": normalized_name, "document_name": normalized_name, "page": page_number}))
        elif extension == ".txt":
            text = _decode_text_bytes(file_bytes)
            if text.strip():
                documents.append(Document(page_content=text, metadata={"source": normalized_name, "document_name": normalized_name, "page": 1, "file_type": "text"}))
        else:
            raise ValueError(f"Unsupported file type: {extension}")

    if not documents:
        raise ValueError("No documents found in the uploaded files")

    return documents


# Backwards-compatible misspelling alias
def load_uploaded_dcouments(uploaded_files: Iterable) -> list[Document]:
    return load_uploaded_documents(uploaded_files)


async def load_uploaded_documents_async(uploaded_files: AsyncIterable) -> list[Document]:
    """Async compatibility wrapper for FastAPI UploadFile objects.

    Accepts an async iterable of UploadFile-like objects (with `.filename` and `.read()`),
    reads their bytes, and delegates to the synchronous loader.
    """
    files = []

    # Support both async iterables (e.g., streaming upload) and regular lists
    # of FastAPI `UploadFile` objects. If `uploaded_files` supports async
    # iteration, use `async for`, otherwise iterate synchronously and await
    # individual `.read()` calls.
    if hasattr(uploaded_files, "__aiter__"):
        async for uploaded in uploaded_files:
            filename = getattr(uploaded, "filename", None) or getattr(uploaded, "name", "")
            content = await uploaded.read()

            class _Tmp:
                pass

            tmp = _Tmp()
            tmp.name = filename
            tmp._content = content

            def getvalue():
                return tmp._content

            tmp.getvalue = getvalue
            files.append(tmp)
    else:
        for uploaded in uploaded_files:
            # uploaded is an UploadFile from FastAPI; its .read() is async,
            # but we're inside an async function so we can await it.
            filename = getattr(uploaded, "filename", None) or getattr(uploaded, "name", "")
            content = await uploaded.read()

            class _Tmp:
                pass

            tmp = _Tmp()
            tmp.name = filename
            tmp._content = content

            def getvalue():
                return tmp._content

            tmp.getvalue = getvalue
            files.append(tmp)

    return await asyncio.to_thread(load_uploaded_documents, files)
