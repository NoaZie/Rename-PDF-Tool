import spacy
from symspellpy import SymSpell, Verbosity
import re
import os

# 📌 Lade das deutsche NLP-Modell von spaCy
try:
    nlp = spacy.load("de_core_news_sm")
except OSError:
    print("[ERROR] spaCy Sprachmodell 'de_core_news_sm' nicht gefunden! Installiere es mit:")
    print("        python -m spacy download de_core_news_sm")
    exit(1)

# 📌 Lade das SymSpell-Modell für automatische Korrekturen
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# 🔍 Wörterbuch-Datei (angepasst für die Umgebung)
DICTIONARY_PATH = "data/german_dictionary.txt"

# Falls das Wörterbuch existiert, lade es
if os.path.exists(DICTIONARY_PATH):
    sym_spell.load_dictionary(DICTIONARY_PATH, term_index=0, count_index=1, separator=" ")
    print(f"[INFO] Wörterbuch erfolgreich geladen: {DICTIONARY_PATH}")
else:
    print(f"[WARNUNG] Wörterbuch nicht gefunden: {DICTIONARY_PATH}")
    print("           OCR-Korrektur wird ohne Wörterbuch durchgeführt!")

def autocorrect_text(text):
    """Nutze SymSpell für automatische Wortkorrekturen."""
    try:
        words = text.split()
        corrected_words = []
        for word in words:
            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            corrected_words.append(suggestions[0].term if suggestions else word)
        return " ".join(corrected_words)
    except Exception as e:
        print(f"[ERROR] Fehler in der Autokorrektur: {e}")
        return text

def clean_ocr_text(text):
    """Entferne fehlerhafte Sonderzeichen & OCR-Artefakte."""
    try:
        text = re.sub(r"[^a-zA-Z0-9äöüÄÖÜß,.!?;:()@/\-]", " ", text)  # Erlaubt nur sinnvolle Zeichen
        text = re.sub(r"\s+", " ", text).strip()  # Entferne doppelte Leerzeichen
        text = re.sub(r"(\d{2})\.(\d{2})\.(\d{2,4})", r"\1. \2. \3", text)  # Datumsformat fixen
        return text
    except Exception as e:
        print(f"[ERROR] Fehler bei der Bereinigung: {e}")
        return text

def improve_sentence_structure(text):
    """Nutze spaCy, um OCR-Text zu bereinigen & Satzumbrüche zu korrigieren."""
    try:
        doc = nlp(text)
        return " ".join([token.text for token in doc])
    except Exception as e:
        print(f"[ERROR] Fehler bei der Satzkorrektur: {e}")
        return text

def restore_paragraphs(text):
    """Versucht, fehlende Absätze zu rekonstruieren, indem Satzstrukturen analysiert werden."""
    try:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])', text)  # Satzenden erkennen, neue Absätze hinzufügen
        structured_text = "\n\n".join(sentences)  # Absätze sinnvoll rekonstruieren
        return structured_text.strip()
    except Exception as e:
        print(f"[ERROR] Fehler bei der Absatzrekonstruktion: {e}")
        return text

def remove_noise(text):
    """Entfernt überflüssige Zeichen und OCR-Artefakte am Anfang & Ende."""
    try:
        text = re.sub(r"^[^a-zA-Z0-9]+", "", text)  # Entfernt Sonderzeichen am Anfang
        text = re.sub(r"[Pp]r f re . f i N u N", "", text)  # Entfernt bekannte Artefakte
        return text.strip()
    except Exception as e:
        print(f"[ERROR] Fehler beim Entfernen von Rauschen: {e}")
        return text

def format_entities(text):
    """Korrigiert die Darstellung von Telefonnummern, E-Mails & Adressen."""
    try:
        text = re.sub(r"(\+?\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{4})", r"\1", text)  # Telefonnummern
        text = re.sub(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", r"\1", text)  # E-Mails
        return text.strip()
    except Exception as e:
        print(f"[ERROR] Fehler beim Formatieren von Entitäten: {e}")
        return text

def correct_ocr_text(text):
    """Führe alle Optimierungen durch: Reinigung, Autokorrektur, Satzstruktur, Absätze & Entitäten."""
    text = clean_ocr_text(text)
    if os.path.exists(DICTIONARY_PATH):
        text = autocorrect_text(text)
    text = improve_sentence_structure(text)
    text = restore_paragraphs(text)
    text = format_entities(text)
    text = remove_noise(text)
    return text

if __name__ == "__main__":
    test_text = "+49 (0)157 7444 5488 05. DEZ. 24 epfaff@the-ip-fox.com"
    corrected_text = correct_ocr_text(test_text)

    print("\n[INFO] Ursprünglicher Text:")
    print(test_text)
    print("\n✅ [INFO] Nach automatischer Korrektur:")
    print(corrected_text)
