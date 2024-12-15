#!/usr/bin/env python3
# rage_dataloader.py

import os
import re
import logging
import requests

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None

class RAGEDataLoader:
    """
    Loads and chunkifies multiple data types: .txt, .md, .pdf, .docx from local paths,
    or plain text from remote URLs.
    """
    def __init__(self, chunk_size=128, allowed_extensions=None):
        if allowed_extensions is None:
            allowed_extensions = ('.txt', '.md', '.pdf', '.docx')
        self.chunk_size = chunk_size
        self.allowed_extensions = allowed_extensions
        logging.basicConfig(level=logging.INFO)

    def load_from_folder(self, folder_path):
        all_chunks = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.allowed_extensions):
                    file_path = os.path.join(root, file)
                    all_chunks.extend(self.load_file(file_path))
        return all_chunks

    def load_file(self, filepath):
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        if ext in ('.txt', '.md'):
            return self._load_text_file(filepath)
        elif ext == '.pdf':
            if PyPDF2 is None:
                logging.error("PyPDF2 not installed. Cannot parse PDF.")
                return []
            return self._load_pdf_file(filepath)
        elif ext == '.docx':
            if docx is None:
                logging.error("python-docx not installed. Cannot parse DOCX.")
                return []
            return self._load_docx_file(filepath)
        else:
            logging.warning(f"Unsupported file type: {ext}")
            return []

    def load_from_url(self, url):
        """
        Fetch text from remote URL using a more “browser-like” user agent 
        to reduce 403 forbidden.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
        try:
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            text = response.text
            return self._chunk_text(text)
        except requests.RequestException as e:
            logging.error(f"Error fetching URL: {url}, {e}")
            return []

    def _load_text_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return self._chunk_text(text)

    def _load_pdf_file(self, filepath):
        chunks = []
        try:
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)
                all_text = []
                for p in range(num_pages):
                    page_obj = reader.pages[p]
                    page_text = page_obj.extract_text() or ""
                    all_text.append(page_text)
                full_text = "\n".join(all_text)
                chunks.extend(self._chunk_text(full_text))
        except Exception as e:
            logging.error(f"Error reading PDF file {filepath}: {e}")
        return chunks

    def _load_docx_file(self, filepath):
        chunks = []
        try:
            doc = docx.Document(filepath)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            all_text = "\n".join(full_text)
            chunks.extend(self._chunk_text(all_text))
        except Exception as e:
            logging.error(f"Error reading DOCX file {filepath}: {e}")
        return chunks

    def _chunk_text(self, text):
        words = re.split(r"\s+", text)
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i:i+self.chunk_size])
            chunks.append(chunk)
        return chunks

