import os
import json
import re
from src.extract_text import extract_text_from_pdf  # Nutzt die vorhandene OCR-Funktion

# Verzeichnis für PDFs
data_dir = "data/pdfs/"  # Stelle sicher, dass deine PDFs hier liegen
output_file = "data/generated_train_data.json"

# Regulärer Ausdruck zum Extrahieren von Absender, Empfänger und Betreff aus dem Dateinamen
filename_pattern = re.compile(r"\d{4}_\d{2}_\d{2}-CS-#-\d{3}-(?P<absender>.+?) an (?P<empfaenger>.+?)-(?P<betreff>.+)\.pdf")

def find_entities(text, filename):
    """Sucht nach Absender, Empfänger und Betreff anhand des Dateinamens und prüft, ob sie im OCR-Text vorkommen."""
    
    filename_pattern = re.compile(r"\d{4}_\d{2}_\d{2}-CS-#-\d{3}-(?P<absender>.+?) an (?P<empfaenger>.+?)-(?P<betreff>.+)\.pdf")

    match = filename_pattern.match(filename)
    if match:
        absender = match.group("absender").replace("_", " ")
        empfaenger = match.group("empfaenger").replace("_", " ")
        betreff = match.group("betreff").replace("_", " ")

        # Entitätenerkennung im OCR-Text testen
        entities = []
        for label, keyword in [("ABSENDER", absender), ("EMPFÄNGER", empfaenger), ("BETREFF", betreff)]:
            position = str(text[1]).find(keyword)  # Greife mit int-Schlüssel auf die erste Seite zu
            if position != -1:
                entities.append((position, position + len(keyword), label))

        return entities
    else:
        print("[ERROR] Dateiname konnte nicht analysiert werden.")
        return []

def create_training_data():
    """Erstellt Trainingsdaten aus allen PDFs im `data/pdfs/`-Ordner."""
    training_data = []

    # Prüfe, ob der Ordner existiert
    if not os.path.exists(data_dir):
        print(f"[ERROR] Das Verzeichnis '{data_dir}' existiert nicht!")
        return

    # Alle PDF-Dateien im Ordner durchgehen
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, filename)
            text = extract_text_from_pdf(pdf_path)

            # Finde die Positionen der Entitäten
            entities = find_entities(text, filename)

            if entities:
                training_data.append({
                    "text": text,
                    "entities": entities
                })
                print(f"[INFO] Entitäten aus {filename} extrahiert.")

    # Speichere die generierten Trainingsdaten
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=4, ensure_ascii=False)
    print(f"[SUCCESS] Trainingsdaten gespeichert: {output_file}")

# Starte die Erstellung der Trainingsdaten
create_training_data()
