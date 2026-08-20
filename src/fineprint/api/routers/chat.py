from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from fineprint.agent.run import ask as default_ask

router = APIRouter()


def get_ask_fn():
    return default_ask


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, ask_fn=Depends(get_ask_fn)) -> ChatResponse:  # noqa: B008 - required FastAPI dependency injection pattern
    try:
        answer = ask_fn(request.question)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return ChatResponse(answer=answer)
