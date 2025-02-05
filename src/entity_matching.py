# entity_matching.py

import re
from rapidfuzz import process, fuzz

def find_best_match(entity_value: str, text: str, threshold: int = 70):
    """
    Sucht den 'besten' unscharfen Treffer (Fuzzy-Match) für entity_value im text.
    Nutzt RapidFuzz (fuzz.ratio) und gibt (start, end) im originalen Text zurück,
    falls der Score >= threshold liegt, sonst None.

    Parameter:
      - entity_value: z. B. "Rechnung 133", "Klähblatt GmbH"
      - text: der gesamte extrahierte PDF-Text
      - threshold: Minimaler Score (0-100), ab dem ein Match akzeptiert wird

    Rückgabe: (start, end) oder None
    """
    if not entity_value or not text:
        return None
    
    # Text z. B. in Zeilen oder Sätze aufteilen, hier simpel: Zeilen
    lines = text.split('\n')
    # Mit extractOne() kriegen wir den besten Match (string, score, index)
    best = process.extractOne(entity_value, lines, scorer=fuzz.ratio)
    
    if best is None:
        return None

    best_string, score, line_idx = best
    
    if score < threshold:
        return None
    
    # Suche best_string nochmals per exakter Suche in text
    match = re.search(re.escape(best_string), text)
    if match:
        return (match.start(), match.end())
    return None
