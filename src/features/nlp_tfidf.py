from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_vectorizer():
    # Word + char n-grams capturam termos técnicos e variações
    return TfidfVectorizer(
        lowercase=False,  # já normalizamos antes
        ngram_range=(1,2),
        min_df=2,
        max_features=50000,
        analyzer="word",
    )

def build_char_vectorizer():
    return TfidfVectorizer(
        lowercase=False,
        analyzer="char",
        ngram_range=(3,5),
        min_df=2,
        max_features=30000,
    )
