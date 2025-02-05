import cv2
import pytesseract
import numpy as np
import pdfplumber
import os
from PIL import Image

# 📌 Falls Tesseract nicht automatisch erkannt wird, hier anpassen:
pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"  # Prüfe mit 'which tesseract'

DATA_FOLDER = "data"
OUTPUT_FOLDER = "data/extracted_text"

def preprocess_image(image):
    """Verbessert die Bildqualität für Tesseract OCR."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
        )
        gray = cv2.medianBlur(gray, 1)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        return gray
    except Exception as e:
        print(f"[ERROR] Fehler bei der Bildvorverarbeitung: {e}")
        return image

def extract_text_from_image(image):
    """Extrahiert Text aus einem Bild mit Tesseract OCR."""
    try:
        processed_image = preprocess_image(image)
        text = pytesseract.image_to_string(processed_image, lang="deu", config="--psm 6")
        return text.strip()
    except Exception as e:
        print(f"[ERROR] Tesseract-OCR-Fehler: {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    """
    Extrahiert Text aus einem gescannten PDF (OCR) oder aus normalem PDF.
    :param pdf_path: Pfad zur PDF-Datei
    :return: Dictionary mit Seitenzahlen und extrahiertem Text
    """
    extracted_text = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                
                if not text:
                    print(f"[INFO] Datei: {pdf_path} - Seite {page_number}: Kein Text gefunden, OCR wird verwendet.")
                    image = page.to_image().original
                    image = np.array(image)
                    text = extract_text_from_image(image)
                
                extracted_text[page_number] = text.strip()
        return extracted_text
    except Exception as e:
        print(f"[ERROR] Fehler bei der Verarbeitung von {pdf_path}: {e}")
        return {}

def process_all_pdfs():
    """Findet und verarbeitet alle PDF-Dateien im `data/`-Ordner."""
    if not os.path.exists(DATA_FOLDER):
        print(f"[ERROR] Ordner '{DATA_FOLDER}' nicht gefunden!")
        return
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("[INFO] Keine PDFs gefunden.")
        return
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_FOLDER, pdf_file)
        print(f"\n🔍 Verarbeite: {pdf_file}")
        
        extracted_text = extract_text_from_pdf(pdf_path)
        
        output_file = os.path.join(OUTPUT_FOLDER, f"{pdf_file}.txt")
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                for page, content in extracted_text.items():
                    f.write(f"Seite {page}:\n{content}\n\n")
            print(f"✅ Datei verarbeitet: {pdf_file} → Text gespeichert in: {output_file}")
        except Exception as e:
            print(f"[ERROR] Fehler beim Speichern von {pdf_file}: {e}")

if __name__ == "__main__":
    try:
        process_all_pdfs()
    except Exception as e:
        print(f"[ERROR] Unerwarteter Fehler: {e}")
