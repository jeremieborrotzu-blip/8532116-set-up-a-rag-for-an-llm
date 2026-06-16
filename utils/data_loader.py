# utils/data_loader.py
import os
import requests
import zipfile
import io
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Text extraction functions (similar to your simple_indexer.py) ---

def extract_text_from_pdf(file_path: str) -> Optional[str]:
    """Extracts the text from a PDF file."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text = "".join(page.extract_text() + "\n" for page in reader.pages if page.extract_text())
        logging.info(f"Text extracted from PDF: {file_path} ({len(text)} characters)")
        return text
    except Exception as e:
        logging.error(f"PDF extraction error {file_path}: {e}")
        return None

def extract_text_from_docx(file_path: str) -> Optional[str]:
    """Extracts the text from a Word DOCX file."""
    try:
        import docx
        doc = docx.Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs if para.text)
        logging.info(f"Text extracted from DOCX: {file_path} ({len(text)} characters)")
        return text
    except Exception as e:
        logging.error(f"DOCX extraction error {file_path}: {e}")
        return None

def extract_text_from_txt(file_path: str) -> Optional[str]:
    """Extracts the text from a plain text file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        logging.info(f"Text extracted from TXT: {file_path} ({len(text)} characters)")
        return text
    except Exception as e:
        logging.error(f"TXT extraction error {file_path}: {e}")
        return None

def extract_text_from_csv(file_path: str) -> Optional[str]:
    """Extracts the text from a CSV file (converts to string)."""
    try:
        import pandas as pd
        # Read with more robust encoding error handling
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1') # Try another common encoding
        except Exception as read_e:
             logging.warning(f"CSV read error {file_path}: {read_e}. Trying with separator ';'")
             try:
                 df = pd.read_csv(file_path, sep=';')
             except UnicodeDecodeError:
                  df = pd.read_csv(file_path, sep=';', encoding='latin1')
             except Exception as read_e2:
                  logging.error(f"Cannot read the CSV {file_path}: {read_e2}")
                  return None

        text = df.to_string()
        logging.info(f"Text extracted from CSV: {file_path} ({len(text)} characters)")
        return text
    except ImportError:
        logging.warning("Pandas not installed. Cannot read CSV files.")
        return None
    except Exception as e:
        logging.error(f"CSV extraction error {file_path}: {e}")
        return None

def extract_text_from_excel(file_path: str) -> Optional[str]:
    """Extracts the text from an Excel file (converts to string)."""
    try:
        import pandas as pd
        df = pd.read_excel(file_path, sheet_name=None) # Read all sheets
        text = ""
        if isinstance(df, dict): # If multiple sheets
             for sheet_name, sheet_df in df.items():
                 text += f"--- Sheet: {sheet_name} ---\n{sheet_df.to_string()}\n\n"
        else: # If a single sheet
             text = df.to_string()
        logging.info(f"Text extracted from Excel: {file_path} ({len(text)} characters)")
        return text
    except ImportError:
        logging.warning("Pandas or openpyxl not installed. Cannot read Excel files.")
        return None
    except Exception as e:
        logging.error(f"Excel extraction error {file_path}: {e}")
        return None

# --- Loading functions ---

def download_and_extract_zip(url: str, output_dir: str) -> bool:
    """Downloads a ZIP file from a URL and extracts it."""
    if not url:
        logging.warning("No URL provided for the download.")
        return False
    try:
        logging.info(f"Downloading the data from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status() # Check for HTTP errors

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True) # Create the output folder

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            logging.info(f"Extracting the content into {output_dir}...")
            z.extractall(output_dir)
        logging.info("Download and extraction completed.")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Download error: {e}")
        return False
    except zipfile.BadZipFile:
        logging.error("The downloaded file is not a valid ZIP.")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during the download/extraction: {e}")
        return False

def load_and_parse_files(input_dir: str) -> List[Dict[str, any]]:
    """
    Recursively loads and parses the files in a directory.
    Returns a list of dictionaries, each representing a document.
    """
    documents = []
    input_path = Path(input_dir)
    if not input_path.is_dir():
        logging.error(f"The input directory '{input_dir}' does not exist.")
        return []

    logging.info(f"Walking the source directory: {input_dir}")
    for file_path in input_path.rglob("*.*"): # Walks all files recursively
        if file_path.is_file():
            relative_path = file_path.relative_to(input_path)
            # The source folder name is the first component of the relative path
            source_folder = relative_path.parts[0] if len(relative_path.parts) > 1 else "root"
            ext = file_path.suffix.lower()
            text = None

            logging.debug(f"Processing file: {relative_path} (Source folder: {source_folder})")

            if ext == ".pdf":
                text = extract_text_from_pdf(str(file_path))
            elif ext == ".docx":
                text = extract_text_from_docx(str(file_path))
            elif ext == ".txt":
                text = extract_text_from_txt(str(file_path))
            elif ext == ".csv":
                text = extract_text_from_csv(str(file_path))
            elif ext in [".xlsx", ".xls"]:
                text = extract_text_from_excel(str(file_path))
            else:
                logging.warning(f"Unsupported file type ignored: {relative_path}")
                continue

            if text:
                documents.append({
                    "page_content": text,
                    "metadata": {
                        "source": str(relative_path), # Relative path as the source
                        "filename": file_path.name,
                        "category": source_folder, # Uses the folder name as category/metadata
                        "full_path": str(file_path.resolve()) # Absolute path if needed
                    }
                })
            else:
                 logging.warning(f"No text could be extracted from {relative_path}")

    logging.info(f"{len(documents)} documents loaded and parsed.")
    return documents
