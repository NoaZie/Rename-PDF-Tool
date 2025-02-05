import os
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np

# Stelle sicher, dass Tesseract richtig konfiguriert ist
pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"
os.environ["TESSDATA_PREFIX"] = "/usr/local/share/tessdata"

# 📌 Pfad zu einer gescannten PDF aus skipped_pdfs (ersetzen mit einem existierenden Pfad!)
pdf_path = "/Users/noah/Desktop/ZVW/Automatisierung/Automatische Umbennenung/nextcloud_pdf_tool/data/OCR test pdf/test.pdf"

def extract_text_with_ocr(pdf_path):
    """Konvertiert eine gescannte PDF in Bilder und extrahiert Text mit OCR."""
    images = convert_from_path(pdf_path)
    extracted_text = ""

    for image in images:
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

        text = pytesseract.image_to_string(thresh, lang="deu")
        extracted_text += text + "\n"

    return extracted_text.strip()

# 📌 OCR-Test starten
ocr_text = extract_text_with_ocr(pdf_path)
print("📜 OCR-Extrahierter Text:\n", ocr_text)
