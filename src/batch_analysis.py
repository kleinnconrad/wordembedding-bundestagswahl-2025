import os
import numpy as np
from gensim.models import Word2Vec

# Import your centralized NLP configuration
from config import get_configured_nlp

# --- Configuration & Pathing ---
PARTIES = ["cdu", "gruene", "spd", "afd"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# --- 30 Political Buzzwords ---
BUZZWORDS = [
    "Klimaschutz", "Innere Sicherheit", "Steuern", "Migration", "Wirtschaft",
    "Digitalisierung", "Bildung", "Rente", "Gesundheit", "Europa",
    "Infrastruktur", "Bürokratie", "Wohnen", "Familie", "Arbeitsplätze",
    "Soziale Gerechtigkeit", "Innovation", "Pflege", "Asyl", "Integration",
    "Verkehr", "Umweltschutz", "Mindestlohn", "Steuerentlastung", "Bundeswehr",
    "Demokratie", "Gleichstellung", "Ländlicher Raum", "Energiewende", "Datenschutz"
]

# Load the pre-configured spaCy model (with all federal stopwords already applied)
print("Loading spaCy model and custom stopwords from config...")
nlp = get_configured_nlp()

def preprocess_query(text):
    """Normalizes input query to lowercase lemmas."""
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

def get_vector_for_phrase(model, tokens):
    """Averages vectors for phrases."""
    valid_vectors = [model.wv[t] for t in tokens if t in model.wv]
    if not valid_vectors:
        return None
    return np.mean(valid_vectors, axis=0)

def run_batch_analysis():
    print(f"\nRunning batch analysis for {len(BUZZWORDS)} buzzwords...\n")

    for party in PARTIES:
        model_path = os.path.join(MODELS_DIR, f"{party}.model")
        if not os.path.exists(model_path):
            print(f"[{party.upper()}] Model not found in {MODELS_DIR}. Skipping.\n")
            continue

        model = Word2Vec.load(model_path)
        
        # Print Table Header
        print(f"{'='*105}")
        print(f" {party.upper()} - Top 3 Semantic Associations")
        print(f"{'='*105}")
        print(f"| {'Buzzword':<22} | {'Match 1 (Dist)':<23} | {'Match 2 (Dist)':<23} | {'Match 3 (Dist)':<23} |")
        print(f"|{'-'*24}|{'-'*25}|{'-'*25}|{'-'*25}|")

        for word in BUZZWORDS:
            query_tokens = preprocess_query(word)
            
            if not query_tokens:
                print(f"| {word:<22} | {'Invalid/Stopword':<75} |")
                continue

            query_vec = get_vector_for_phrase(model, query_tokens)

            if query_vec is None:
                print(f"| {word:<22} | {'Not in Vocabulary':<75} |")
                continue

            # Fetch extra results to allow filtering out the query terms
            raw_results = model.wv.most_similar(positive=[query_vec], topn=15)
            filtered_results = [(w, sim) for w, sim in raw_results if w not in query_tokens][:3]

            # Format the output row
            cols = []
            for match_word, similarity in filtered_results:
                distance = 1 - similarity
                cols.append(f"{match_word} ({distance:.3f})")
            
            # Fill empty slots if less than 3 matches were found (highly unlikely but safe)
            while len(cols) < 3:
                cols.append("N/A")

            print(f"| {word:<22} | {cols[0]:<23} | {cols[1]:<23} | {cols[2]:<23} |")
            
        print(f"{'='*105}\n")

if __name__ == "__main__":
    run_batch_analysis()