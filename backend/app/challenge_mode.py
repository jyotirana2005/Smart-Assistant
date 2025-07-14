import random, nltk, re, uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import app.qa_engine as qa_engine

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

LATEST_QS: list[dict] = []
VECT = TfidfVectorizer(stop_words="english")

QUESTION_PATTERNS = [
    (r"used in (.+)", "In which fields is AI used?"),
    (r"known for (.+)", "What is it known for?"),
    (r"consists of (.+)", "What does it consist of?"),
    (r"includes (.+)", "What does it include?"),
    (r"benefits include (.+)", "What are the benefits?"),
    (r"helps (.+)", "What does it help with?"),
]

def _make_comprehension_question(chunk: str) -> dict:
    sentences = [s.strip() for s in nltk.sent_tokenize(chunk) if len(s.strip()) > 30]
    if not sentences:
        return {}

    sent = random.choice(sentences)
    # Try to remove a key phrase for a fill-in-the-blank style question
    words = sent.split()
    if len(words) > 8:
        blank_index = random.randint(3, len(words) - 3)
        blanked = words[:blank_index] + ["_____"] + words[blank_index + 1:]
        question = f"Fill in the blank: {' '.join(blanked)}"
    else:
        question = f"What is the meaning of: \"{sent}\""

    return {
        "id": str(uuid.uuid4()),
        "type": "comprehension",
        "question": question,
        "answer": sent,
        "context": chunk.strip()
    }



def generate_challenge(n: int = 3):
    chunks = [c for c in qa_engine._text_chunks if c.strip()]
    if not chunks:
        return [{"type": "info", "question": "Upload document first."}]
    LATEST_QS.clear()
    for _ in range(n):
        chunk = random.choice(chunks)
        LATEST_QS.append(_make_comprehension_question(chunk))
    return LATEST_QS

def evaluate_single_answer(question_id: str, user_answer: str):
    print("[🧪] Evaluating against question ID:", question_id)
    print("[🧪] Current LATEST_QS IDs:", [q["id"] for q in LATEST_QS])

    for q in LATEST_QS:
        if q["id"] == question_id:  # ✅ GOOD!
            gold = q["answer"]
            break
    else:
        return {
            "result": "incorrect",
            "feedback": "⚠️ Question not found."
        }

    # ...similarity check...


    try:
        vecs = VECT.fit_transform([user_answer, gold])
        sim = cosine_similarity(vecs[0], vecs[1]).item()
    except Exception as e:
        return {
            "result": "incorrect",
            "feedback": f"❌ Similarity computation failed: {str(e)}"
        }

    if sim > 0.75:
        result = "correct"
    elif sim > 0.4:
        result = "partial"
    else:
        result = "incorrect"

    return {
        "result": result,
        "feedback": f"Similarity: {sim:.2f}. Reference: {gold}"
    }
