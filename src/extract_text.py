import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path
import os
import cv2
import numpy as np
from ocr_preprocessing import preprocess_image
from ocr_postprocessing import correct_ocr_text

def extract_text_from_pdf(pdf_path):
    """Versucht, Text direkt aus dem PDF zu extrahieren. Falls nicht möglich, wird OCR verwendet."""
    try:
        text = ""
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text += page.get_text("text")
        
        if text.strip():
            return correct_ocr_text(text)  # Direkte Textextraktion erfolgreich mit Nachbearbeitung
        
        # Falls kein Text vorhanden ist, OCR verwenden
        print(f"⚠️ Kein eingebetteter Text gefunden in {pdf_path}, starte OCR...")
        return extract_text_with_ocr(pdf_path)
    except Exception as e:
        print(f"Fehler beim Extrahieren des Textes aus {pdf_path}: {e}")
        return ""

def extract_text_with_ocr(pdf_path):
    """Extrahiert Text aus einem gescannten PDF mit OCR, nutzt Preprocessing."""
    try:
        images = convert_from_path(pdf_path)
        extracted_text = ""

        for image in images:
            if image is not None:
                # Stelle sicher, dass das Bild im richtigen Format ist
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  
                gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                
                extracted_text += pytesseract.image_to_string(gray, lang="deu") + "\n"
            else:
                print(f"⚠️ Fehler: Das Bild aus {pdf_path} konnte nicht geladen werden!")
                return ""

        return extracted_text.strip()
    
    except Exception as e:
        print(f"Fehler bei der OCR-Verarbeitung von {pdf_path}: {e}")
        return ""

def save_extracted_text(pdf_path, output_folder):
    """Speichert den extrahierten Text in einer Datei."""
    try:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        text = extract_text_from_pdf(pdf_path)
        output_file = output_folder / f"{Path(pdf_path).stem}.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        
        return output_file
    except Exception as e:
        print(f"Fehler beim Speichern der extrahierten Daten für {pdf_path}: {e}")
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python extract_text.py <pdf_path> <output_folder>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_folder = sys.argv[2]
    
    output_file = save_extracted_text(pdf_path, output_folder)
    if output_file:
        print(f"Extracted text saved to {output_file}")
