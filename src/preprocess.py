import os
from gensim.models import Word2Vec

# Import your centralized NLP configuration
from config import get_configured_nlp

# --- Configuration & Pathing ---
PARTIES = ["cdu", "gruene", "spd", "afd"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# --- Manual Overrides ---
# Leave empty {} to auto-generate based on corpus size.
# Example: "gruene": {"epochs": 3, "vector_size": 120}
MANUAL_OVERRIDES = {
    "cdu": {},
    "gruene": {}, 
    "spd": {},
    "afd": {}
}

# Load the pre-configured spaCy model (with all federal stopwords already applied)
print("Loading spaCy model and custom stopwords from config...")
nlp = get_configured_nlp()

def preprocess_text(text):
    """
    Lemmatizes text and removes stop words, punctuation, and short noise.
    """
    doc = nlp(text)
    stop_words = nlp.Defaults.stop_words
    
    tokens = []
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma not in stop_words and token.text.lower() not in stop_words:
            if not token.is_punct and not token.is_space and token.is_alpha:
                if len(lemma) > 1:
                    tokens.append(lemma)
    return tokens

def calculate_hyperparameters(total_tokens):
    """
    Dynamically generates Word2Vec hyperparameters based on token count.
    Prevents the 'Curse of Dimensionality' on small corpora and overfitting on large ones.
    """
    # Base configuration common to all sizes
    params = {
        "window": 5,
        "workers": 4,
        "sg": 1  # Skip-gram
    }

    # Small Corpus (e.g., highly condensed programs)
    if total_tokens < 15000:
        params["vector_size"] = 60      # Lower dimensions prevent sparsity
        params["epochs"] = 12           # More passes needed to learn weights
        params["min_count"] = 2         # Keep rarer words
        params["tier"] = "Small"

    # Medium Corpus
    elif total_tokens < 30000:
        params["vector_size"] = 100     # Standard dimensionality
        params["epochs"] = 7            # Moderate training passes
        params["min_count"] = 2
        params["tier"] = "Medium"

    # Large Corpus (e.g., Grüne Langfassung)
    else:
        params["vector_size"] = 120     # Richer vector space possible
        params["epochs"] = 4            # Fewer passes prevent overfitting
        params["min_count"] = 3         # Drop rare noise, enough data exists
        params["tier"] = "Large"

    return params

def run_preprocessing_and_training():
    if not os.path.exists(MODELS_DIR):
        print(f"Creating directory: {MODELS_DIR}")
        os.makedirs(MODELS_DIR)

    for party in PARTIES:
        file_path = os.path.join(DATA_DIR, f"{party}.txt")
        if not os.path.exists(file_path):
            print(f"Skipping {party}: file not found at {file_path}")
            continue

        print(f"\n{'='*40}")
        print(f" Processing: {party.upper()}")
        print(f"{'='*40}")
        
        processed_sentences = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                clean_line = preprocess_text(line)
                if clean_line:
                    processed_sentences.append(clean_line)
        
        total_tokens = sum(len(s) for s in processed_sentences)
        
        # 1. Generate auto parameters based on size
        auto_params = calculate_hyperparameters(total_tokens)
        corpus_tier = auto_params.pop("tier") # Remove tier before passing to Word2Vec
        
        # 2. Fetch manual overrides for this specific party
        manual_params = MANUAL_OVERRIDES.get(party, {})
        
        # 3. Merge parameters (Manual overrides overwrite auto parameters)
        final_params = {**auto_params, **manual_params}

        print(f"Corpus Metrics:")
        print(f"  - Unique context windows: {len(processed_sentences)}")
        print(f"  - Total clean tokens:     {total_tokens} ({corpus_tier} Tier)")
        print(f"Hyperparameters:")
        for key, value in final_params.items():
            # Highlight if the value was manually overridden
            is_overridden = " (Manual Override)" if key in manual_params else ""
            print(f"  - {key:<12}: {value}{is_overridden}")

        # Training the Word2Vec model using dictionary unpacking
        model = Word2Vec(sentences=processed_sentences, **final_params)
        
        model_save_path = os.path.join(MODELS_DIR, f"{party}.model")
        model.save(model_save_path)
        print(f"\n✅ Model saved to {model_save_path}")

if __name__ == "__main__":
    run_preprocessing_and_training()