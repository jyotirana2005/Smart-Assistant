from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
import traceback, re, json    
from fastapi import APIRouter, Form



import app.qa_engine as qa_engine
from app.doc_parser import parse_document
from app.summarizer import summarize_text
from app.challenge_mode import generate_challenge
from app.challenge_mode import evaluate_single_answer

router = APIRouter()


# ───────────────────────── Global state ──────────────────────────
parsed_text: str = ""          # stores current document text

# regex to strip lone surrogate code‑points (invalid UTF‑8)
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
def strip_surrogates(s: str) -> str:
    return _SURROGATE_RE.sub("", s)

# ───────────────────────── FastAPI setup ─────────────────────────
app = FastAPI(title="Offline Basic Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  # you had llow_credentials=True
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────── Endpoints ───────────────────────────

@app.get("/")
def home():
    return {"message": "Smart Research Assistant API is running"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    
    try:
        global parsed_text
        text = await parse_document(file)
        if not text.strip():
            raise ValueError("No extractable text found in document.")

        parsed_text = text                 # save for other routes
        qa_engine.init(text)               # build index

        print("len(_text_chunks) after init:", len(qa_engine._text_chunks))
        
        summary = strip_surrogates(summarize_text(text))
        return {"summary": summary}

    except Exception as e:
        traceback.print_exc()
        return PlainTextResponse(str(e), status_code=500)

@app.post("/ask")
async def ask(question: str = Form(...)):
    """
    Answer a question using the indexed document chunks.
    """
    try:
        if not parsed_text.strip():
            return JSONResponse(
                {"error": "Please upload a document first."}, status_code=400
            )

        # ensure index persists (helpful when hot‑reload is off)
        if qa_engine._matrix is None:      # ← _matrix is the vector store
            qa_engine.init(parsed_text)

        answer, citation = qa_engine.answer_question(question)
        return JSONResponse(
            {"answer": strip_surrogates(answer), "citation": strip_surrogates(citation)}
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

from app.challenge_mode import generate_challenge

@app.post("/challenge/start")
def challenge_start():
    qs = generate_challenge()
    safe = json.loads(strip_surrogates(json.dumps(qs, ensure_ascii=False)))
    return JSONResponse(content={"questions": safe})


@app.post("/challenge/evaluate")
async def challenge_evaluate(
    question_id: str = Form(...),
    answer: str = Form(...)
):
    return evaluate_single_answer(question_id, answer)




