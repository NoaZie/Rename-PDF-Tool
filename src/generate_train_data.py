import json
import os
import argparse

# 📌 CLI-Argument für den Ordner
parser = argparse.ArgumentParser(description="Erstelle Trainingsdaten aus PDFs in einem Ordner")
parser.add_argument("folder", type=str, help="Pfad zum Ordner mit PDFs")
args = parser.parse_args()

DATA_FOLDER = args.folder  
OUTPUT_PATH = os.path.join(DATA_FOLDER, "generated_train_data.json")

if not os.path.exists(DATA_FOLDER):
    print(f"❌ Fehler: Ordner '{DATA_FOLDER}' nicht gefunden!")
    exit(1)

pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]
TRAIN_DATA = []

for filename in pdf_files:
    parts = filename.replace(".pdf", "").split("-")

    if len(parts) < 6:
        print(f"⚠ Datei {filename} hat unerwartete Struktur, wird übersprungen.")
        continue

    absender = parts[-3].strip()
    empfänger = parts[-2].strip()
    betreff = parts[-1].strip()

    TRAIN_DATA.append((filename, {"entities": [
        (filename.index(absender), filename.index(absender) + len(absender), "ABSENDER"),
        (filename.index(empfänger), filename.index(empfänger) + len(empfänger), "EMPFÄNGER"),
        (filename.index(betreff), filename.index(betreff) + len(betreff), "BETREFF")
    ]}))

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(TRAIN_DATA, f, indent=4, ensure_ascii=False)

print(f"✅ Trainingsdaten gespeichert: {OUTPUT_PATH}")
