import spacy
nlp = spacy.load("de_core_news_sm")
print(nlp.Defaults.stop_words)