import spacy

try:
    nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])
except OSError:
    print("Error: German spaCy model not found.")
    print("Please run: python -m spacy download de_core_news_sm")
    exit(1)

# --- FEDERAL CUSTOM STOP WORDS (Bundestagswahl) ---
# Replaced state-level terms (BW, Land) with federal equivalents (Bund, Deutschland).
# Added federal parties (CSU, FDP) and specific program terms.
custom_stop_words = {
    "deutschland", "bund", "bundesregierung", "bundestag", "regierung",
    "wahlprogramm", "regierungsprogramm", "kapitel", "seite", "bzw", "sowie", "dabei",
    "cdu", "csu", "union", "grüne", "spd", "afd", "fdp", 
    "müssen", "wollen", "sollen", "setzen", "werden",
    "unser", "unserer", "unseren", "jahr", "jahre", "ab"
}

# Register the custom words into the spaCy vocabulary
for word in custom_stop_words:
    nlp.Defaults.stop_words.add(word)
    nlp.vocab[word].is_stop = True

def preprocess_text(text):
    """
    Lemmatizes text and removes stop words, punctuation, and short noise.
    Uses a strict manual check against the stopword list to bypass casing bugs.
    """
    doc = nlp(text)
    stop_words = nlp.Defaults.stop_words
    
    tokens = []
    for token in doc:
        lemma = token.lemma_.lower()
        # Filter by stop words (strict manual check)
        if lemma not in stop_words and token.text.lower() not in stop_words:
            # Filter punctuation, spaces, and non-alphabetic characters
            if not token.is_punct and not token.is_space and token.is_alpha:
                # Exclude single-character noise
                if len(lemma) > 1:
                    tokens.append(lemma)
                    
    return tokens

# --- Verification ---
print(f"Total stop words in model: {len(nlp.Defaults.stop_words)}")
print(f"Is 'deutschland' recognized? {'deutschland' in nlp.Defaults.stop_words}")
print(f"Is 'bundestag' recognized? {'bundestag' in nlp.Defaults.stop_words}")

# Quick test string
test_sentence = "Die CDU und die CSU wollen in Deutschland den Bundestag stärken."
print(f"Raw: {test_sentence}")
print(f"Cleaned: {preprocess_text(test_sentence)}")