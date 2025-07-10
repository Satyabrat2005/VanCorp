from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import upload, tryon

app = FastAPI(
    title = "Virtual Try-On API",
    description = "Next-gen AI-powered virtual fashion try-on system",
    version = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/upload", tags=["Upload"]) # type: ignore
app.include_router(tryon.router, prefix="/tryon", tags=["Try-On"]) # type: ignore

@app.get("/")
def root():
    return {"message": "✅ VanCorp AI Virtual Try-On API is running!"}
