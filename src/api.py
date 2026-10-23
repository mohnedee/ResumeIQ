
from pathlib import Path
import logging
from io import BytesIO
from uuid import uuid4

import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from src.auth_routes import router as auth_router
from src.database import init_db, SessionLocal, User, Resume


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)
MODEL_PATH = BASE_DIR / "models" / "best_model_calibrated.joblib"
SECRET_PATH = BASE_DIR / ".session_secret"

if not SECRET_PATH.exists():
    raise RuntimeError("Session secret file is missing.")

SESSION_SECRET = SECRET_PATH.read_text().strip()

if len(SESSION_SECRET) < 32:
    raise RuntimeError("Session secret is invalid.")

model = joblib.load(MODEL_PATH)
init_db()


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="ResumeIQ",
    description="NLP-based resume classification and HR management system.",
    version="1.0.0",
)

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",
    https_only=False,
    max_age=3600,
)

app.include_router(auth_router)


# --------------------------------------------------
# Request models
# --------------------------------------------------

class ResumeRequest(BaseModel):
    resume_text: str = Field(
        min_length=20,
        max_length=50000,
        description="The text extracted from a resume",
    )


# --------------------------------------------------
# Database dependency
# --------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------
# Basic API routes
# --------------------------------------------------

@app.get("/")
def home():
    return {"message": "ResumeIQ API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


# --------------------------------------------------
# Classify resume text
# --------------------------------------------------

@app.post("/predict")
def predict_resume(request: ResumeRequest):
    text = request.resume_text.strip()

    if len(text) < 20:
        raise HTTPException(
            status_code=400,
            detail="Resume text must contain at least 20 characters.",
        )

    prediction = str(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]
    class_index = list(model.classes_).index(prediction)
    confidence = float(probabilities[class_index]) * 100
    logger.info("Text resume classified: category=%s", prediction)

    return {
        "predicted_category": prediction,
        "confidence": round(confidence, 2),
        "message": "Resume classified successfully",
    }


# --------------------------------------------------
# Upload, classify, and save a PDF resume
# --------------------------------------------------

@app.post("/predict-pdf")
async def predict_resume_pdf(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    user_id = request.session.get("user_id")

    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail="Please log in first.",
        )

    user = db.get(User, user_id)

    if user is None:
        request.session.clear()
        raise HTTPException(
            status_code=401,
            detail="User account not found.",
        )

    if user.role != "candidate":
        raise HTTPException(
            status_code=403,
            detail="Candidate account required.",
        )

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    content = await file.read(5 * 1024 * 1024 + 1)

    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="PDF files must be 5 MB or smaller."
        )

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid PDF file."
        )

    try:
        reader = PdfReader(BytesIO(content))
        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        ).strip()
    except Exception:
        logger.exception("Failed to extract text from uploaded PDF.")
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded PDF.",
        )

    if len(text) < 20:
        raise HTTPException(
            status_code=400,
            detail="The PDF contains too little extractable text.",
        )

    prediction = str(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]
    class_index = list(model.classes_).index(prediction)
    confidence = float(probabilities[class_index]) * 100

    upload_dir = BASE_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    stored_filename = f"{uuid4().hex}.pdf"
    stored_path = upload_dir / stored_filename

    stored_path.write_bytes(content)

    try:
        resume = Resume(
            user_id=user.id,
            original_filename=Path(
                file.filename or "resume.pdf"
            ).name,
            stored_filename=stored_filename,
            extracted_text=text,
            predicted_category=prediction,
            confidence=confidence,
        )

        db.add(resume)
        db.commit()
        db.refresh(resume)
        logger.info(
            "Resume uploaded and classified: resume_id=%s, user_id=%s, category=%s",
            resume.id,
            user.id,
            prediction,
        )

    except Exception:
        logger.exception("Failed to save resume: user_id=%s", user.id)
        db.rollback()
        stored_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="Could not save the resume. Please try again.",
        )

    return {
        "message": "PDF resume classified and saved successfully",
        "resume_id": resume.id,
        "predicted_category": prediction,
        "confidence": round(confidence, 2),
        "extracted_text": text,
    }


# --------------------------------------------------
# Get the current candidate's resumes only
# --------------------------------------------------

@app.get("/my-resumes")
def get_my_resumes(
    request: Request,
    db: Session = Depends(get_db),
):
    user_id = request.session.get("user_id")

    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail="Please log in first.",
        )

    user = db.get(User, user_id)

    if user is None:
        request.session.clear()
        raise HTTPException(
            status_code=401,
            detail="User account not found.",
        )

    if user.role != "candidate":
        raise HTTPException(
            status_code=403,
            detail="Candidate account required.",
        )

    resumes = (
        db.query(Resume)
        .filter(Resume.user_id == user.id)
        .order_by(Resume.uploaded_at.desc())
        .all()
    )

    return {
        "resumes": [
            {
                "id": resume.id,
                "filename": resume.original_filename,
                "category": resume.predicted_category,
                "confidence": resume.confidence,
                "uploaded_at": (
                    resume.uploaded_at.isoformat()
                    if resume.uploaded_at
                    else None
                ),
            }
            for resume in resumes
        ]
    }


# --------------------------------------------------
# HR: View all candidate resumes
# --------------------------------------------------

@app.get("/hr/resumes")
def get_all_resumes_for_hr(
    request: Request,
    db: Session = Depends(get_db),
):
    user_id = request.session.get("user_id")

    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail="Please log in first.",
        )

    user = db.get(User, user_id)

    if user is None:
        request.session.clear()
        raise HTTPException(
            status_code=401,
            detail="User account not found.",
        )

    if user.role != "hr":
        raise HTTPException(
            status_code=403,
            detail="HR account required.",
        )

    resumes = (
        db.query(Resume, User)
        .join(User, Resume.user_id == User.id)
        .order_by(Resume.uploaded_at.desc())
        .all()
    )

    return {
        "resumes": [
            {
                "id": resume.id,
                "filename": resume.original_filename,
                "candidate_name": (
                    f"{candidate.first_name} "
                    f"{candidate.last_name}"
                ),
                "candidate_email": candidate.email,
                "category": resume.predicted_category,
                "confidence": resume.confidence,
                "uploaded_at": (
                    resume.uploaded_at.isoformat()
                    if resume.uploaded_at
                    else None
                ),
            }
            for resume, candidate in resumes
        ]
    }


@app.get("/hr/resumes/{resume_id}/download")
def download_resume_for_hr(
    resume_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    user_id = request.session.get("user_id")

    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail="Please log in first.",
        )

    user = db.get(User, user_id)

    if user is None:
        request.session.clear()
        raise HTTPException(
            status_code=401,
            detail="User account not found.",
        )

    if user.role != "hr":
        raise HTTPException(
            status_code=403,
            detail="HR account required.",
        )

    resume = db.get(Resume, resume_id)

    if resume is None:
        raise HTTPException(
            status_code=404,
            detail="Resume not found.",
        )

    upload_dir = (BASE_DIR / "uploads").resolve()
    stored_name = Path(resume.stored_filename).name
    stored_path = (upload_dir / stored_name).resolve()

    if stored_path.parent != upload_dir or not stored_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Resume file not found.",
        )

    logger.info(
        "Resume downloaded by HR: resume_id=%s, hr_user_id=%s",
        resume.id,
        user.id,
    )

    return FileResponse(
        path=stored_path,
        media_type="application/pdf",
        filename=resume.original_filename,
    )


# --------------------------------------------------
# Web application
# --------------------------------------------------

@app.get("/app", response_class=FileResponse)
def web_app():
    return FileResponse(
        BASE_DIR / "templates" / "index.html"
    )


# --------------------------------------------------
# Static files
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)
