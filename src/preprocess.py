import os
import spacy
from gensim.models import Word2Vec

# --- Configuration & Pathing ---
PARTIES = ["cdu", "gruene", "spd", "afd"]

# Robust pathing: dynamically finds your project root 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Load spaCy with the German model
try:
    nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])
except OSError:
    print("Error: Run 'python -m spacy download de_core_news_sm' first.")
    exit(1)

# --- FEDERAL CUSTOM STOP WORDS (Bundestagswahl) ---
custom_stop_words = {
    "deutschland", "bund", "bundesregierung", "bundestag", "regierung",
    "wahlprogramm", "regierungsprogramm", "kapitel", "seite", "bzw", "sowie", "dabei",
    "cdu", "csu", "union", "grüne", "spd", "afd", "fdp", 
    "müssen", "wollen", "sollen", "setzen", "werden",
    "unser", "unserer", "unseren", "jahr", "jahre", "ab"
}

for word in custom_stop_words:
    nlp.Defaults.stop_words.add(word)
    nlp.vocab[word].is_stop = True

def preprocess_text(text):
    """
    Lemmatizes text and removes stop words, punctuation, and short noise.
    """
    doc = nlp(text)
    stop_words = nlp.Defaults.stop_words
    
    tokens = []
    for token in doc:
        lemma = token.lemma_.lower()
        # Filter by stop words (manual check), punctuation, and length
        if lemma not in stop_words and token.text.lower() not in stop_words:
            if not token.is_punct and not token.is_space and token.is_alpha:
                if len(lemma) > 1:
                    tokens.append(lemma)
    return tokens

def run_preprocessing_and_training():
    if not os.path.exists(MODELS_DIR):
        print(f"Creating directory: {MODELS_DIR}")
        os.makedirs(MODELS_DIR)

    for party in PARTIES:
        file_path = os.path.join(DATA_DIR, f"{party}.txt")
        if not os.path.exists(file_path):
            print(f"Skipping {party}: file not found at {file_path}")
            continue

        print(f"--- Processing {party.upper()} ---")
        
        # We store the document as a list of 'sentences' (token lists)
        processed_sentences = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                clean_line = preprocess_text(line)
                if clean_line:
                    processed_sentences.append(clean_line)
        
        total_tokens = sum(len(s) for s in processed_sentences)
        print(f"Total unique context windows: {len(processed_sentences)}")
        print(f"Total tokens after cleaning: {total_tokens}")

        # Determine epochs dynamically based on corpus size
        current_epochs = 4 if party == "gruene" else 6
        print(f"Training Word2Vec model for {current_epochs} epochs...")

        # Training the Word2Vec model
        # Using Skip-gram (sg=1) is often better for smaller, high-quality datasets
        model = Word2Vec(
            sentences=processed_sentences,
            vector_size=100,
            window=5,
            min_count=2,
            workers=4,
            sg=1,
            epochs=current_epochs  
        )
        
        model_save_path = os.path.join(MODELS_DIR, f"{party}.model")
        model.save(model_save_path)
        print(f"✅ Model saved to {model_save_path}\n")

if __name__ == "__main__":
    run_preprocessing_and_training()