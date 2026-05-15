# Political Word Embeddings: Election Programs

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/kleinnconrad/wordembedding-landtagswahl-bw-2026/tree/main)

This project utilizes natural language processing and Word2Vec models to analyze the semantic context of political election programs. 

## Training Note
Please note that due to a significant difference in the underlying text corpus size, the word embedding model for the **Grüne** (Greens) election program was trained with a different number of epochs compared to the other parties. This adjustment prevents overfitting or underfitting, ensuring the vector space accurately reflects their specific semantic landscape.

## Further Information
For a detailed breakdown of the theoretical framework, preprocessing pipeline, and how to use the CLI query tool to find contextual synonyms, please visit the main repository linked above.