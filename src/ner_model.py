import spacy
from spacy.training import Example
from spacy.tokens import DocBin
import random

# Lade das deutsche Modell mit NER-Funktion
nlp = spacy.load("de_core_news_sm")

# Trainingsdaten für Named Entity Recognition (NER)
train_data = [
    ("Rechnung Nr. 233074341 von Behn an Klähblatt GmbH",
     {"entities": [(22, 26, "BETREFF"), (30, 34, "ABSENDER"), (38, 53, "EMPFÄNGER")]}),

    ("Zahlungsnachweis Lohnsteuer 11.2023 - Ali Düsmez an Klähblatt GmbH",
     {"entities": [(0, 27, "BETREFF"), (30, 39, "ABSENDER"), (43, 58, "EMPFÄNGER")]}),

    ("Adobe Systems an ZvW Beteiligung GmbH - Rechnung IEN204007186794",
     {"entities": [(0, 13, "ABSENDER"), (17, 37, "EMPFÄNGER"), (41, 70, "BETREFF")]}),
]

# Vorbereitung des Trainings-Datasets
db = DocBin()
for text, annotations in train_data:
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in annotations["entities"]:
        span = doc.char_span(start, end, label=label)
        if span is not None:
            ents.append(span)
    doc.ents = ents
    db.add(doc)

# Speichere die Trainingsdaten für spätere Nutzung
train_data_path = "data/ner_training_data.spacy"
db.to_disk(train_data_path)

print("✅ Trainingsdaten gespeichert. Das Modell kann jetzt trainiert werden!")

# Modell trainieren
ner = nlp.get_pipe("ner")

# Neue Labels hinzufügen
for _, annotations in train_data:
    for ent in annotations["entities"]:
        ner.add_label(ent[2])

# Training vorbereiten
optimizer = nlp.resume_training()
random.shuffle(train_data)
losses = {}

print("🚀 Starte Training...")
for epoch in range(10):  # 10 Durchläufe für das Training
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.2, losses=losses)

    print(f"🔄 Epoch {epoch+1} abgeschlossen, Verlust: {losses}")

# Trainiertes Modell speichern
nlp.to_disk("data/ner_model")

print("✅ Modell gespeichert! Du kannst es jetzt laden & testen.")

# Test: Modell auf einem neuen Text anwenden
test_text = "Rechnung Nr. 243005710 von Behn an Klähblatt GmbH"
doc = nlp(test_text)
extracted_entities = [(ent.text, ent.label_) for ent in doc.ents]

print("\n🎯 Extrahierte Entitäten:", extracted_entities)
