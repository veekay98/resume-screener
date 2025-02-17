from fastapi import FastAPI, UploadFile, File, HTTPException
from mangum import Mangum
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os
import warnings
import logging

# ✅ Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Set Hugging Face Cache Path
os.environ["HF_HOME"] = "/tmp/huggingface"

# ✅ Load Model & FAISS at Startup (Cache in /tmp)
MODEL_PATH = "/tmp/sentence-transformers"

if not os.path.exists(MODEL_PATH):
    print("Downloading Model...")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODEL_PATH)
    print("Model Downloaded!")
else:
    print("Loading Model from /tmp...")
    embed_model = SentenceTransformer(MODEL_PATH)
print("Model Ready!")

# ✅ Initialize FAISS Index
index = faiss.IndexFlatL2(384)  # FAISS index with 384-dimensional embeddings
resumes = []  # Store resume texts
resume_embeddings = []  # Store resume embeddings

# **Utility Functions**

def extract_text_from_pdf(file_path):
    """Extracts text from a given PDF file."""
    with pdfplumber.open(file_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

def embed_text(text):
    """Embeds text using Sentence Transformers."""
    return embed_model.encode(text)

# **API Endpoints**

# ✅ Health Check Endpoint
@app.get("/")
async def health_check():
    return {"status": "OK"}

# ✅ Lambda Test Route
@app.get("/test")
async def test():
    return {"message": "Lambda is working!"}

# ✅ Upload Resume & Store in FAISS
@app.post("/upload_resume/")
async def upload_resume(file: UploadFile = File(...)):
    """Uploads a resume (PDF), extracts text, and stores it in FAISS."""
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract text from the PDF
    extracted_text = extract_text_from_pdf(file_path)

    # Store in FAISS Index
    resumes.append(extracted_text)
    resume_embedding = embed_text(extracted_text).reshape(1, -1)
    index.add(resume_embedding)

    return {"filename": file.filename, "text": extracted_text[:500]}

# ✅ Retrieve All Uploaded Resumes
@app.get("/list_resumes/")
async def list_resumes():
    """Lists all stored resumes."""
    if not resumes:
        return {"message": "No resumes uploaded yet."}
    return {"resumes": resumes}

# ✅ Find Best Resume Match for a Job Description
@app.post("/match_resume/")
async def match_resume(job_description: str):
    """Finds the best matching resume for a given job description."""
    if not job_description:
        raise HTTPException(status_code=400, detail="Job description is required")

    try:
        job_embedding = embed_text(job_description).reshape(1, -1)
        D, I = index.search(job_embedding, k=1)

        best_match_index = int(I[0][0])
        best_match_resume = resumes[best_match_index]
        return {"best_match_resume": best_match_resume, "similarity_score": float(D[0][0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Retrieve Resume by Index
@app.get("/get_resume/{resume_index}")
async def get_resume(resume_index: int):
    """Retrieves a specific resume by its index."""
    if resume_index < 0 or resume_index >= len(resumes):
        raise HTTPException(status_code=404, detail="Resume not found")
    return {"resume": resumes[resume_index]}

# ✅ AI-Powered Resume Refinement
@app.post("/refine_match/")
async def refine_match(job_text: str, resume_text: str):
    """Uses OpenAI (GPT-4) to refine the match between a resume and job description."""
    if not job_text or not resume_text:
        raise HTTPException(status_code=400, detail="Both job_text and resume_text are required")

    try:
        chat_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Match the resume with the job description"},
                {"role": "user", "content": f"Resume: {resume_text}, Job: {job_text}"},
            ],
            max_tokens=500,
            temperature=0.5,
        )

        return {"refined_match": chat_response.choices[0].message.content.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Ensure AWS Lambda API Gateway Triggers FastAPI
handler = Mangum(app)
