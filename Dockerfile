# ✅ Use Python 3.9 Slim as Base Image
FROM python:3.9-slim

WORKDIR /app

# ✅ Install dependencies
RUN apt-get update && apt-get install -y python3-dev git wget gcc curl \
    && rm -rf /var/lib/apt/lists/*

# ✅ Copy Application Files into the Container
COPY . .

# ✅ Install Python Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Set Hugging Face Cache
ENV HF_HOME=/tmp/huggingface

# ✅ Preload Model & FAISS (To Reduce Cold Start Times)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/tmp/sentence-transformers')"
RUN python -c "import faiss; index = faiss.IndexFlatL2(384)"

# ✅ AWS Lambda does NOT need Uvicorn
CMD ["python", "main.py"]
