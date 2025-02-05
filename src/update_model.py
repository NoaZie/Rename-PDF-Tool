import spacy
import json
import os

DATA_FOLDER = "data"
MODEL_PATH = os.path.join(DATA_FOLDER, "ner_model")
CORRECTION_LOG = os.path.join(DATA_FOLDER, "correction_logs.json")

if not os.path.exists(MODEL_PATH):
    print("❌ Fehler: Modell nicht gefunden!")
    exit(1)

nlp = spacy.load(MODEL_PATH)

if not os.path.exists(CORRECTION_LOG):
    print("❌ Keine neuen Korrekturen gefunden. Training wird übersprungen.")
    exit(0)

with open(CORRECTION_LOG, "r", encoding="utf-8") as f:
    try:
        CORRECTIONS = json.load(f)
    except json.JSONDecodeError:
        print("⚠️ Fehler beim Laden der Korrekturen. Datei könnte beschädigt sein.")
        exit(1)

if not CORRECTIONS:
    print("ℹ️ Keine neuen Korrekturen vorhanden.")
    exit(0)

TRAIN_DATA = []
for correction in CORRECTIONS:
    text = correction["text"]
    entities = []
    
    if "absender" in correction and correction["absender"]["text"]:
        start = text.find(correction["absender"]["text"])
        if start != -1:
            entities.append((start, start + len(correction["absender"]["text"]), "ABSENDER"))
    
    if "empfänger" in correction and correction["empfänger"]["text"]:
        start = text.find(correction["empfänger"]["text"])
        if start != -1:
            entities.append((start, start + len(correction["empfänger"]["text"]), "EMPFÄNGER"))
    
    if "betreff" in correction and correction["betreff"]["text"]:
        start = text.find(correction["betreff"]["text"])
        if start != -1:
            entities.append((start, start + len(correction["betreff"]["text"]), "BETREFF"))
    
    if entities:
        TRAIN_DATA.append((text, {"entities": entities}))

if not TRAIN_DATA:
    print("ℹ️ Keine validen Trainingsdaten vorhanden. Beende Skript.")
    exit(0)

nlp.disable_pipes("ner")
optimizer = nlp.resume_training()

for epoch in range(5):
    losses = {}
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = spacy.training.Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.3, losses=losses)
    print(f"Epoch {epoch + 1} abgeschlossen. Verlust: {losses}")

nlp.to_disk(MODEL_PATH)
print("✅ Training abgeschlossen! Das Modell wurde aktualisiert.")

# Nach dem Training die Korrekturen löschen, um doppelte Verarbeitung zu vermeiden
with open(CORRECTION_LOG, "w", encoding="utf-8") as f:
    json.dump([], f)
