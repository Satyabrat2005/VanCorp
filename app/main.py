# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router

app = FastAPI(
    title="Vanity AI Virtual Try-On",
    description="AI-powered try-on backend using deep learning and NLP",
    version="1.0.0"
)

# Allow all origins for testing (configure in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes from /api/routes.py
app.include_router(api_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to Vanity AI Virtual Try-On API 🚀"}
