"""
SHL Assessment Recommender — FastAPI Application

Endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → Conversational assessment recommendations
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import ChatRequest, ChatResponse, HealthResponse
from agent import get_agent, SHLAgent


# ── Lifespan: warm up retriever + agent on startup ─────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the retriever and agent on startup."""
    print("🚀 Starting SHL Assessment Recommender...")
    start = time.time()
    try:
        agent = get_agent()
        elapsed = time.time() - start
        print(f"✅ Agent ready in {elapsed:.1f}s "
              f"({len(agent.retriever.catalog)} catalog items)")
    except Exception as e:
        print(f"⚠️ Startup warning: {e}")
        print("   The agent will initialize on first request.")
    yield
    print("👋 Shutting down SHL Assessment Recommender.")


# ── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "A conversational agent that recommends SHL Individual Test Solutions "
        "based on hiring needs through multi-turn dialogue."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for flexibility during evaluation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns status 'ok' when the service is ready.",
)
async def health():
    """Health check endpoint. Returns HTTP 200 with {"status": "ok"}."""
    return HealthResponse(status="ok")


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with the SHL Assessment Advisor",
    description=(
        "Stateless endpoint. Send the full conversation history and "
        "receive the agent's next reply with optional recommendations."
    ),
)
async def chat(request: ChatRequest):
    """
    Process a conversation and return the agent's response.
    
    The request must contain the full conversation history.
    The response includes:
      - reply: natural language response
      - recommendations: null or 1-10 assessment items
      - end_of_conversation: true when the task is complete
    """
    try:
        agent = get_agent()
        response = await agent.chat(request.messages)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Chat error: {e}")
        # Return a graceful error response instead of 500
        return ChatResponse(
            reply="I apologize, but I encountered an error processing your request. Please try again.",
            recommendations=None,
            end_of_conversation=False,
        )


# ── Run with uvicorn ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
