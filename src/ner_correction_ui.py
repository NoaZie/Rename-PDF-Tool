import streamlit as st
import spacy
import json
import os
import shutil
import re
from extract_text import extract_text_from_pdf
from entity_matching import find_best_match
from annotated_text import annotated_text

# Verzeichnisse definieren
PDF_FOLDER = "data/pdfs/"
PROCESSED_FOLDER = "data/processed_pdfs/"
FAILED_FOLDER = "data/failed_pdfs/"
SKIPPED_FOLDER = "data/skipped_pdfs/"
TRAINING_LOG = "data/training_log.json"
ERROR_LOG = "data/error_log.json"
CORRECTION_LOG = "data/correction_logs.json"
MODEL_PATH = "data/ner_model"

# Stelle sicher, dass alle Verzeichnisse existieren
for folder in [PDF_FOLDER, PROCESSED_FOLDER, FAILED_FOLDER, SKIPPED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Lade das trainierte NER-Modell
nlp = spacy.load(MODEL_PATH)

def log_training(data):
    """Speichert die Trainingsdaten sicher und überprüft JSON-Integrität."""
    try:
        if not os.path.exists(TRAINING_LOG):
            with open(TRAINING_LOG, "w", encoding="utf-8") as f:
                json.dump([], f)
        
        with open(TRAINING_LOG, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
                if not isinstance(logs, list):
                    logs = []
            except json.JSONDecodeError:
                logs = []
        
        logs.append(data)
        
        with open(TRAINING_LOG, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)
    except Exception as e:
        st.error(f"Fehler beim Speichern der Trainingsdaten: {e}")

def extract_entities_from_filename(filename):
    match = re.match(r"(\d{4}_\d{2}_\d{2})-.*?#-\d+-(.*?) an (.*?)-(.*)\.pdf", filename)
    if match:
        return {"absender": match.group(2), "empfänger": match.group(3), "betreff": match.group(4)}
    return None

def save_correction(original_text, absender, empfaenger, betreff, pdf_path):
    correction_data = {
        "text": original_text,
        "absender": absender,
        "empfänger": empfaenger,
        "betreff": betreff
    }
    if os.path.exists(CORRECTION_LOG):
        with open(CORRECTION_LOG, "r", encoding="utf-8") as f:
            try:
                corrections = json.load(f)
                if not isinstance(corrections, list):
                    corrections = []
            except json.JSONDecodeError:
                corrections = []
    else:
        corrections = []
    corrections.append(correction_data)
    with open(CORRECTION_LOG, "w", encoding="utf-8") as f:
        json.dump(corrections, f, indent=4, ensure_ascii=False)
    log_training(correction_data)
    shutil.move(pdf_path, os.path.join(PROCESSED_FOLDER, os.path.basename(pdf_path)))
    st.session_state.clear()
    st.rerun()

def handle_failed_pdf(pdf_path):
    shutil.move(pdf_path, os.path.join(FAILED_FOLDER, os.path.basename(pdf_path)))
    st.session_state.clear()
    st.rerun()

def skip_pdf(pdf_path):
    shutil.move(pdf_path, os.path.join(SKIPPED_FOLDER, os.path.basename(pdf_path)))
    st.session_state.clear()
    st.rerun()

def get_next_pdf():
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    return os.path.join(PDF_FOLDER, pdf_files[0]) if pdf_files else None

def get_correction_count():
    if os.path.exists(CORRECTION_LOG):
        with open(CORRECTION_LOG, "r", encoding="utf-8") as f:
            try:
                corrections = json.load(f)
                return len(corrections)
            except json.JSONDecodeError:
                return 0
    return 0

st.title("KI-gestützte PDF-Verarbeitung")
st.write("Das System lädt PDFs automatisch aus dem Ordner und erkennt relevante Informationen.")

correction_count = get_correction_count()
st.sidebar.write(f"Anzahl gespeicherter Korrekturen: **{correction_count}**")
if correction_count >= 50:
    st.sidebar.warning("Zeit für ein Training! Mindestens 50 Korrekturen erreicht.")

pdf_path = get_next_pdf()
if pdf_path:
    text_input = extract_text_from_pdf(pdf_path)
    filename_entities = extract_entities_from_filename(os.path.basename(pdf_path)) or {}
    
    if "manuelle_entitaeten" not in st.session_state:
        st.session_state.manuelle_entitaeten = {}
    
    for entity_type, entity_value in st.session_state.manuelle_entitaeten.items():
        filename_entities[entity_type] = filename_entities.get(entity_type, "") + (", " + entity_value if entity_value else "")
    
    colors = {"empfänger": "blue", "absender": "green", "betreff": "yellow"}
    highlighted_text = []
    words = text_input.split()
    
    for word in words:
        found = False
        for entity_type, entity_value in filename_entities.items():
            if word in entity_value.split():
                highlighted_text.append((f"**{word}**", entity_type.capitalize(), colors.get(entity_type, "black")))
                found = True
                break
        if not found:
            highlighted_text.append((word, "", "black"))
    
    annotated_text(*highlighted_text)
    
    st.subheader("Manuelle Entitätserkennung")
    selected_text = st.text_input("Gib eine vollständige Entität aus dem Text ein")
    entity_type = st.selectbox("Wähle die Entitätskategorie", ["Absender", "Empfänger", "Betreff"])
    
    if st.button("Entität hinzufügen") and selected_text:
        st.session_state.manuelle_entitaeten[entity_type.lower()] = selected_text
        st.rerun()
    
    st.subheader("Erkannte Werte")
    with st.form("correction_form"):
        absender = st.text_input("Absender", filename_entities.get("absender", ""))
        empfaenger = st.text_input("Empfänger", filename_entities.get("empfänger", ""))
        betreff = st.text_input("Betreff", filename_entities.get("betreff", ""))
        submitted = st.form_submit_button("Korrektur speichern")
        if submitted:
            save_correction(text_input, absender, empfaenger, betreff, pdf_path)
    
    if st.button("PDF ignorieren"):
        skip_pdf(pdf_path)
else:
    st.info("Bitte ein neues PDF in den Ordner 'data/pdfs/' legen.")
