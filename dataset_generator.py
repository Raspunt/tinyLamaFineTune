import json
import re
import os
import PyPDF2
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetGenerator:

    def __init__(self) -> None:
        self.dataset_folder_path = "./datasets"
        self.pdf_raw_datasets_folder = f"{self.dataset_folder_path}/pdf"
        self.json_datasets_folder = f"{self.dataset_folder_path}/json"
        self.init_datasets()

    def init_datasets(self):
        self.create_dataset_folder()

    def create_dataset_folder(self):
        try:
            os.makedirs(self.json_datasets_folder, exist_ok=True)
            os.makedirs(self.pdf_raw_datasets_folder, exist_ok=True)
        except Exception as e:
            logging.error(f"Error creating directories: {e}")

    def extract_text_from_pdf(self, pdf_path: str) -> list:
        """Extracts and validates lines of text from a PDF file."""
        extracted_data = []
        try:
            reader = PyPDF2.PdfReader(pdf_path)
            for page in tqdm(reader.pages, desc=f"Processing {os.path.basename(pdf_path)}"):
                text = page.extract_text()
                lines = filter(None, text.split("\n"))
                extracted_data.extend(
                    {"text": line} for line in lines if self.validate_line(line)
                )
        except Exception as e:
            logging.error(f"Error extracting text from {pdf_path}: {e}")
        return extracted_data

    def create_json_dataset_1book(self, book_path: str) -> str:
        """Processes a single PDF file and saves its content to a JSON file."""
        try:
            if not book_path.endswith(".pdf"):
                logging.error("Provided file is not a PDF.")
                return

            if not os.path.exists(book_path):
                logging.error(f"File does not exist: {book_path}")
                return

            all_data = self.extract_text_from_pdf(book_path)

            if all_data:
                json_path = os.path.join(
                    self.json_datasets_folder, f"{os.path.basename(book_path)}.json")
                with open(json_path, "w", encoding="utf-8") as json_file:
                    json.dump(all_data, json_file, ensure_ascii=False, indent=4)
                logging.info(f"Dataset saved: {json_path}")

                return json_path
            else:
                logging.warning("No valid data extracted from the PDF.")

        except Exception as e:
            logging.error(f"Error processing book: {e}")

    def create_json_dataset_all_books(self):
        """Processes all PDFs in the folder and combines their data into one JSON file."""
        try:
            if not os.path.exists(self.pdf_raw_datasets_folder):
                logging.error(
                    f"PDF folder does not exist: {self.pdf_raw_datasets_folder}")
                return

            pdf_files = [
                f for f in os.listdir(self.pdf_raw_datasets_folder) if f.endswith(".pdf")
            ]

            if not pdf_files:
                logging.warning("No PDF files found in the folder.")
                return

            all_data = []
            for pdf_file in tqdm(pdf_files, desc="Processing all PDFs"):
                pdf_path = os.path.join(self.pdf_raw_datasets_folder, pdf_file)
                all_data.extend(self.extract_text_from_pdf(pdf_path))

            if all_data:
                json_path = os.path.join(
                    self.json_datasets_folder, "dataset.json")
                with open(json_path, "w", encoding="utf-8") as json_file:
                    json.dump(all_data, json_file, ensure_ascii=False, indent=4)
                logging.info(f"All data saved to {json_path}")
                logging.info(f"Count of lines written: {len(all_data)}")
            else:
                logging.warning("No valid data extracted from any PDFs.")

        except Exception as e:
            logging.error(f"Error creating JSON datasets: {e}")

    def validate_line(self, line: str) -> bool:
        """Validates if a line of text should be included."""
        cleaned_line = re.sub(r"[^A-Za-zА-Яа-я0-9\s]", "", line).strip()
        return bool(cleaned_line) and not cleaned_line.isdigit()
