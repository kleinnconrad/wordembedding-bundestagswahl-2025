# Knowing what is meant! Knowing the context!

## Overview
This repository focuses on developing a core capability in the "machine understanding" of political language. When analyzing election programs, the primary objective is to move beyond simple keyword counting and literal text interpretation to genuinely understand **what a political party actually means** when they use specific terminology. This project serves as a dedicated pipeline to explore the semantic landscapes of the 2025 Bundestags election.

---

## The Core Philosophy: Meaning Over Words
Traditionally, political text analysis relies on manually curated lists of keywords or simple frequency counts (e.g., "How often did the CDU mention 'Wirtschaft'?"). This approach is fundamentally limited because it depends entirely on human assumption and misses the surrounding context.

By applying word embeddings, we are challenging this paradigm by asking:
* Can a machine discover related political concepts and thematic priorities that human analysts might overlook?
* Does the machine possess a unique, data-driven "perspective" on a party's ideology based purely on their sentence structure?

### The "Anti-Thesaurus" Principle
It is crucial to understand what this system is *not*. **The model is more than just a simple thesaurus, and it definitely shouldn't act like one.** The goal of this repository is not to find literal word-for-word replacements. Instead, the focus is on understanding the surrounding semantic intent and the context in which words are used. When the SPD and the AfD both mention "Klimaschutz," they do not mean the same thing. By mapping these terms contextually, we reveal the *actual* intent behind the political buzzwords.

---

## Methodology: Discovering Meaning through Vectors
To uncover what a party might mean, we utilize an artificial neural network (Word2Vec) trained independently on each party's text corpus.

1.  **Vector Generation:** The model generates a mathematical vector for every relevant term in the election program. This vector serves as a numerical representation of the term's context within that specific party's platform.
2.  **Contextual Similarity:** We calculate the angle between these "word vectors" (Cosine Distance). Terms that appear in similar or identical contexts will have a smaller angle between their vectors. The smaller the angle, the more similar the context.

---

## The Analytical Pipeline
Instead of manually guessing synonyms, we use the machine to surface contextual relationships through a three-step process:

### 1. Training the Perspective
We train separate continuous bag-of-words/skip-gram models on the raw text of each party (CDU, Grüne, SPD, AfD). This ensures the machine learns four distinct "perspectives" without cross-contamination.

### 2. Identification
Using the built-in Query CLI, we apply the trained models to specific political concepts (e.g., "Innere Sicherheit"). The system automatically identifies "context-similar" terms based on vector proximity, showing us exactly what concepts live in the same neighborhood as our query.

### 3. Evaluation & Interpretation
The machine-generated associations are then evaluated using semantic distance bins (from "Core Identity" to "Semantic Noise"). The ultimate measure of success for this tool is its ability to provide human analysts with empirical data to support political science conclusions—proving ideological divergence through mathematics rather than just subjective reading.