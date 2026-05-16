import spacy

def get_configured_nlp():
    nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])
    
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
        
    return nlp