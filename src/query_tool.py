import os
import argparse
import spacy
import numpy as np
from gensim.models import Word2Vec

# --- Configuration & Pathing ---
PARTIES = ["cdu", "gruene", "spd", "afd"]

# Get the absolute path of the current script (src/query_tool.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up one level to the project root to find 'models/'
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Load spaCy for German lemmatization
try:
    nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])
except OSError:
    print("Error: German spaCy model 'de_core_news_sm' not found.")
    print("Please run: python -m spacy download de_core_news_sm")
    exit(1)

# Custom stop words (should match your training script exactly)
custom_stop_words = {
    "baden-württemberg", "bw", "land", "landesregierung", 
    "wahlprogramm", "kapitel", "seite", "bzw", "sowie", "dabei",
    "cdu", "grüne", "spd", "afd", "müssen", "wollen", "sollen", "setzen",
    "unser", "unserer", "unseren", "jahr", "jahre"
}
for word in custom_stop_words:
    nlp.Defaults.stop_words.add(word)

def get_semantic_label(distance):
    """
    Categorizes the strength of the association based on cosine distance.
    Useful for qualitative analysis in a bachelor thesis.
    """
    if distance <= 0.25:
        return "Core Identity"
    elif distance <= 0.45:
        return "Strong Association"
    elif distance <= 0.65:
        return "Thematic Context"
    elif distance <= 0.80:
        return "Weak/Incidental"
    else:
        return "Semantic Noise"

def preprocess_query(text):
    """
    Normalizes input query to lowercase lemmas and filters stop words.
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

def get_vector_for_phrase(model, tokens):
    """
    Averages vectors for phrases to allow multi-word synonym searches.
    """
    # Fetch vectors only for tokens that actually exist in the model's vocabulary
    valid_vectors = [model.wv[t] for t in tokens if t in model.wv]
    
    if not valid_vectors:
        return None
    
    # Return the centroid (mean) of the phrase tokens
    return np.mean(valid_vectors, axis=0)

def run_query(phrase):
    """
    Loads party models and prints the top 5 synonyms ranked by cosine distance,
    including semantic labels for better interpretation.
    """
    query_tokens = preprocess_query(phrase)
    
    if not query_tokens:
        print(f"Error: No valid words found in query '{phrase}' after cleaning.")
        return

    print(f"\nAnalyzing semantic associations for: '{' '.join(query_tokens)}'")
    print("=" * 80)

    for party in PARTIES:
        model_path = os.path.join(MODELS_DIR, f"{party}.model")
        
        if not os.path.exists(model_path):
            print(f"[{party.upper()}] Model file not found.")
            continue

        model = Word2Vec.load(model_path)
        query_vec = get_vector_for_phrase(model, query_tokens)

        if query_vec is None:
            print(f"[{party.upper()}] Vocabulary match failed for this query.")
            print("-" * 80)
            continue

        # Fetch more results initially to allow for filtering the search term itself
        raw_results = model.wv.most_similar(positive=[query_vec], topn=15)
        
        # Filter: Remove words that are part of the original query
        filtered_results = [
            (word, sim) for word, sim in raw_results 
            if word not in query_tokens
        ][:5]

        print(f"Top 5 for {party.upper()}:")
        if not filtered_results:
            print("  - No distinct associations found.")
        else:
            for word, similarity in filtered_results:
                # Cosine Distance = 1 - Cosine Similarity
                distance = 1 - similarity
                label = get_semantic_label(distance)
                
                # Formatted output for readability
                print(f"  - {word:<22} | Dist: {distance:.4f} | Label: [{label}]")
        
        print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Political WordEmbedding Query Tool")
    parser.add_argument("phrase", type=str, help="Phrase to search (e.g., 'Innere Sicherheit')")
    
    args = parser.parse_args()
    run_query(args.phrase)