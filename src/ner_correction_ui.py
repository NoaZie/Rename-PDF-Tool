import streamlit as st
import spacy
import json
import os
import shutil
import re
from extract_text import extract_text_from_pdf

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

# Lade das trainierte NER-Modell (falls benötigt)
nlp = spacy.load(MODEL_PATH)

def log_training(data):
    """Speichert Trainingsdaten sicher in TRAINING_LOG (Listenformat)."""
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
    """
    Erwartet ein Muster wie:
      2024_11_13-CS-#-133-Dropscan an ZvW Beteiligungen GmbH-Rechnung 24111351.pdf

    match.group(2) => Absender
    match.group(3) => Empfänger
    match.group(4) => Betreff
    """
    match = re.match(
        r"(\d{4}_\d{2}_\d{2})-.*?#-\d+-(.*?) an (.*?)-(.*)\.pdf",
        filename
    )
    if match:
        return {
            "absender": match.group(2),
            "empfänger": match.group(3),
            "betreff": match.group(4)
        }
    return {}

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
    if pdf_files:
        return os.path.join(PDF_FOLDER, pdf_files[0])
    return None

def get_correction_count():
    if os.path.exists(CORRECTION_LOG):
        with open(CORRECTION_LOG, "r", encoding="utf-8") as f:
            try:
                corrections = json.load(f)
                return len(corrections)
            except json.JSONDecodeError:
                return 0
    return 0

def save_correction(original_text, filename_entities, manual_annotations, pdf_path):
    """
    Speichert die finalen Daten:
      - original_text: Vollständiger PDF-Text
      - filename_entities: Dict (Absender, Empfänger, Betreff)
      - manual_annotations: Liste von Dicts ({start, end, label, substring})
    """
    entities = []
    for ann in manual_annotations:
        label = ann["label"]
        substring = ann["substring"]
        soll_text = filename_entities.get(label.lower(), "")
        
        entity_data = {
            "start": ann["start"],
            "end": ann["end"],
            "label": label.upper(),
            "soll": soll_text,
            "ist": substring
        }
        entities.append(entity_data)

    # Trainingsdatensatz
    training_data = {
        "text": original_text,
        "entities": entities,
        "filename_entities": filename_entities
    }

    # Correction-Datensatz (alte Logik)
    correction_record = {
        "text": original_text,
        "filename_entities": filename_entities,
        "manual_entities": manual_annotations
    }
    
    # In correction_logs.json anhängen
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
    corrections.append(correction_record)
    
    with open(CORRECTION_LOG, "w", encoding="utf-8") as f:
        json.dump(corrections, f, indent=4, ensure_ascii=False)
    
    # In TRAINING_LOG
    log_training(training_data)
    
    # PDF verschieben -> processed_pdfs
    shutil.move(pdf_path, os.path.join(PROCESSED_FOLDER, os.path.basename(pdf_path)))
    st.session_state.clear()
    st.rerun()

def find_all_occurrences(text, substring):
    """Alle Start/End-Offets für exakte Vorkommen von 'substring' in 'text'."""
    pattern = re.escape(substring)
    return [(m.start(), m.end()) for m in re.finditer(pattern, text)]

def highlight_text_with_annotations(text, annotations):
    """
    Hebt bestimmte Textstellen je nach Entität farbig hervor.
    Hintergrund ist eine kräftige Farbe, Text bleibt weiß (#FFF),
    damit es sich vom typischen dunklen Layout abhebt.
    """
    # Helle, kräftige Farben mit weißer Schrift:
    entity_colors = {
        "Absender":  "background-color:#FF4500; color:#FFFFFF;",  # Orange-Rot
        "Empfänger": "background-color:#32CD32; color:#FFFFFF;",  # LimeGreen
        "Betreff":   "background-color:#1E90FF; color:#FFFFFF;"   # DodgerBlue
    }
    
    # Sortiere nach Start-Index
    sorted_anns = sorted(annotations, key=lambda x: x["start"])
    
    highlighted = ""
    last_pos = 0
    for ann in sorted_anns:
        start, end = ann["start"], ann["end"]
        label = ann["label"]  # z. B. "Absender", "Empfänger", "Betreff"
        snippet = text[start:end]
        
        # Wähle das Style-Fragment passend zum Label
        style = entity_colors.get(label, "background-color:#555555; color:#FFFFFF;")
        
        # Unmarkierter Text
        highlighted += text[last_pos:start]
        # Markierter Abschnitt
        highlighted += f"<span style='{style}'>{snippet}</span>"
        last_pos = end
    
    # Rest vom Text anhängen
    highlighted += text[last_pos:]
    return highlighted


# -------------------------------------
#   Streamlit-UI
# -------------------------------------
st.title("KI-gestützte PDF-Verarbeitung")
st.write("Dieses System lädt PDFs automatisch aus dem Ordner und erkennt relevante Informationen.")

# Zähle die bisherigen Korrekturen
count = get_correction_count()
st.sidebar.write(f"Anzahl gespeicherter Korrekturen: **{count}**")
if count >= 50:
    st.sidebar.warning("Zeit für ein Training! Mindestens 50 Korrekturen erreicht.")

pdf_path = get_next_pdf()
if pdf_path:
    # OCR nur einmal durchführen pro PDF
    if "current_pdf" not in st.session_state or st.session_state["current_pdf"] != pdf_path:
        st.session_state["current_pdf"] = pdf_path
        st.session_state["manual_annotations"] = []
        st.session_state["ocr_done"] = False

    if not st.session_state.get("ocr_done", False):
        # OCR/Extraktion -> nur einmal
        extracted_text = extract_text_from_pdf(pdf_path)
        if not extracted_text:
            handle_failed_pdf(pdf_path)
        else:
            st.session_state["extracted_text"] = extracted_text
            st.session_state["ocr_done"] = True
    else:
        extracted_text = st.session_state["extracted_text"]

    # "Soll"-Werte aus Dateiname
    filename_entities = extract_entities_from_filename(os.path.basename(pdf_path))

    # Dateiname anzeigen
    st.subheader(f"Dateiname: {os.path.basename(pdf_path)}")

    # Button "Automatische Markierung" -> exakte Vorkommen Absender/Empfänger/Betreff
    if st.button("Automatische Markierung"):
        for key in ["absender", "empfänger", "betreff"]:
            soll_string = filename_entities.get(key, "")
            if soll_string:
                occs = find_all_occurrences(extracted_text, soll_string)
                for (start, end) in occs:
                    st.session_state["manual_annotations"].append({
                        "start": start,
                        "end": end,
                        "label": key.capitalize(),
                        "substring": extracted_text[start:end]
                    })
        st.rerun()

    # Immer nur EINE Anzeige: 
    # "extracted_text" plus vorhandene Markierungen
    st.subheader("Aktueller Text mit Markierungen")
    highlighted = highlight_text_with_annotations(extracted_text, st.session_state["manual_annotations"])
    st.markdown(f"<div style='white-space: pre-wrap;'>{highlighted}</div>", unsafe_allow_html=True)

    # Manuelle Eingabe
    st.subheader("Manuelle Entitätserkennung")
    manual_str = st.text_input("Gib eine vollständige Entität (mehrere Wörter) ein:")
    manual_label = st.selectbox("Wähle die Entitätskategorie", ["Absender", "Empfänger", "Betreff"])
    if st.button("Manuell hinzufügen"):
        if manual_str.strip():
            found = find_all_occurrences(extracted_text, manual_str.strip())
            if not found:
                st.warning("Keine Stelle im Text gefunden (prüfe Groß-/Kleinschreibung).")
            else:
                for (start, end) in found:
                    st.session_state["manual_annotations"].append({
                        "start": start,
                        "end": end,
                        "label": manual_label,
                        "substring": extracted_text[start:end]
                    })
                st.rerun()
        else:
            st.warning("Bitte eine Entität eingeben.")

    st.subheader("Soll-Werte (aus Dateiname) & Korrektur")
    abs_soll = st.text_input("Absender (Soll)", filename_entities.get("absender", ""))
    emp_soll = st.text_input("Empfänger (Soll)", filename_entities.get("empfänger", ""))
    betr_soll = st.text_input("Betreff (Soll)", filename_entities.get("betreff", ""))

    with st.form("save_form"):
        submitted = st.form_submit_button("Korrektur speichern (Training)")
        if submitted:
            filename_entities["absender"] = abs_soll
            filename_entities["empfänger"] = emp_soll
            filename_entities["betreff"] = betr_soll
            
            save_correction(
                original_text=extracted_text,
                filename_entities=filename_entities,
                manual_annotations=st.session_state["manual_annotations"],
                pdf_path=pdf_path
            )

    if st.button("PDF ignorieren"):
        skip_pdf(pdf_path)

else:
    st.info("Bitte ein neues PDF in den Ordner 'data/pdfs/' legen.")
