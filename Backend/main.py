from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings


app = FastAPI(
    title="Reasearch Paper RAG API",
    description="AI-powered Research Paper Retrieval-Augmented Generation API",
    version="1.0.0",
    debug=settings.debug
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Startup event
@app.on_event("startup")
async def startup_event():
    print("\n" + "="*50)
    print("Research Paper RAG API Starting...")
    print("="*50)
    print(f"Environment: {settings.environment}")
    print(f"Debug Mode: {settings.debug}")
    print(f"CORS Origins: {settings.cors_origins}")
    print(f"Upload Directory: {settings.upload_dir}")
    print(f"ChromaDB Directory: {settings.chroma_persist_dir}")
    print("="*50 + "\n")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    print("\n Shutting down Research Paper RAG API...\n")


# Root endpoint
@app.get("/")
def read_root():
    """Root endpoint - API information"""
    return {
        "name": "Research Paper RAG API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "environment": settings.environment,
        "models": {
            "chat": settings.chat_model,
            "embedding": settings.embedding_model
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development
    )