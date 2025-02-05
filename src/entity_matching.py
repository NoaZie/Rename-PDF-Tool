import spacy
from rapidfuzz import fuzz

# Lade das deutsche spaCy-Modell (mittlere Größe für Wort-Embeddings)
nlp = spacy.load("de_core_news_md")

def find_entity_fuzzy(query, text, threshold=80):
    """
    Vergleicht eine Entität mit dem Text und prüft, ob eine fuzzy-Übereinstimmung vorhanden ist.
    :param query: Der zu suchende Begriff (z. B. Absendername)
    :param text: Der gesamte Text, in dem gesucht wird
    :param threshold: Ähnlichkeitsschwelle (80% Standard)
    :return: Gefundene Position oder None
    """
    words = text.split()
    best_match = None
    best_score = 0

    for word in words:
        score = fuzz.ratio(query.lower(), word.lower())
        if score > best_score:
            best_score = score
            best_match = (text.find(word), text.find(word) + len(word))

    return best_match if best_score >= threshold else None

def find_entity_semantic(query, text):
    """
    Nutzt spaCy-Embeddings, um semantisch ähnliche Begriffe zu finden.
    :param query: Der zu suchende Begriff
    :param text: Der gesamte Text
    :return: Beste gefundene Übereinstimmung (Start- und Endposition)
    """
    doc = nlp(text)
    best_match = None
    best_score = 0.0

    for token in doc:
        score = token.similarity(nlp(query))
        if score > best_score:
            best_score = score
            best_match = (token.idx, token.idx + len(token.text))

    return best_match if best_score > 0.7 else None  # Schwelle für semantische Ähnlichkeit

def find_entity(query, text):
    """
    Sucht eine Entität im Text mit folgenden Methoden:
    1. Exakte Übereinstimmung
    2. Fuzzy Matching (80% Schwelle)
    3. Wort-Embeddings (Semantische Ähnlichkeit)
    :return: Position im Text (Start, Ende) oder None
    """
    # 1️⃣ Exakte Übereinstimmung
    if query in text:
        start_idx = text.find(query)
        return start_idx, start_idx + len(query)

    # 2️⃣ Fuzzy Matching
    fuzzy_result = fuzzy_match(query, text)
    if fuzzy_result:
        return fuzzy_result

    # 3️⃣ Embeddings für semantische Suche
    embedding_result = embedding_match(query, text)
    if embedding_result:
        return embedding_result

    return None  # Falls keine Methode erfolgreich war

def find_best_match(query, text):
    return find_entity(query, text)
