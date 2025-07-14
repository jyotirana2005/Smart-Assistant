import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

STOPWORDS = list(nltk.corpus.stopwords.words("english"))

def summarize_text(text: str, n_sentences: int = 20) -> str:
    """Return top‑n sentences based on TF‑IDF importance, nicely formatted without HTML."""
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= n_sentences:
        return "\n\n".join(s.strip() for s in sentences if s.strip())

    vectorizer = TfidfVectorizer(stop_words=STOPWORDS)
    tfidf = vectorizer.fit_transform(sentences)

    scores = tfidf.sum(axis=1).A1
    top_idx = np.argsort(scores)[-n_sentences:][::-1]
    top_sentences = [sentences[i].strip() for i in sorted(top_idx)]

    formatted = []
    for s in top_sentences:
        s = s.strip()
        # Bullet or numbered list
        if re.match(r"^[-•●▪️🔹*•●]|^\d+\.", s):
            formatted.append(f"📌 {s}")
        # Short heading lines
        elif len(s) <= 60:
            formatted.append(f"\n🟣 {s.upper()}\n")
        else:  # Regular paragraph
            formatted.append(s)

    return "\n\n".join(formatted)
