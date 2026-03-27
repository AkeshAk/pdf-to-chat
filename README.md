# Tamil Doc Chatbot

Chat with 4000 Tamil PDFs in Tamil or English.

## Stack
- **Frontend**: HTML/CSS/JS → Vercel (free)
- **Backend**: FastAPI → Render (free)
- **Vector DB**: Qdrant Cloud (free tier or Railway ~₹830/month)
- **LLM**: Groq llama-3.1-8b-instant (free)
- **Embeddings**: paraphrase-multilingual-MiniLM-L12-v2 (Tamil + English)

---

## Setup Steps

### 1. Get API Keys
- **Groq**: https://console.groq.com → API Keys → Create
- **Qdrant Cloud**: https://cloud.qdrant.io → Create Cluster → Get URL + API Key

### 2. Configure environment
Edit `backend/.env`:
```
GROQ_API_KEY=gsk_xxxx
QDRANT_URL=https://xxxx.qdrant.io
QDRANT_API_KEY=xxxx
COLLECTION_NAME=tamil_docs
```

### 3. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 4. Ingest PDFs
Place all PDFs in the `pdfs/` folder, then run:
```bash
cd backend
python ingest.py
```
This will take time for 4000 PDFs. Run once.

### 5. Test locally
```bash
cd backend
uvicorn main:app --reload
```
Open http://localhost:8000

### 6. Deploy Backend to Render
1. Push project to GitHub
2. Go to https://render.com → New Web Service
3. Connect your GitHub repo
4. Set root directory to `backend`
5. Add environment variables from `.env`
6. Deploy

### 7. Deploy Frontend to Vercel
1. Edit `frontend/app.js` → set `BACKEND_URL` to your Render URL
2. Go to https://vercel.com → New Project
3. Import your GitHub repo
4. Set root directory to `frontend`
5. Deploy

---

## Cost Summary
| Service | Cost |
|---|---|
| Groq | Free |
| Qdrant Cloud (1GB) | Free (up to ~200k vectors) |
| Render | Free |
| Vercel | Free |
| **Total** | **₹0 - ₹830/month** |
