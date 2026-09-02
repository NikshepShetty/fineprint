from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Literal
from slowapi import Limiter
from slowapi.util import get_remote_address

from fineprint.agent.run import ask as default_ask

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


def get_ask_fn():
    return default_ask


class ChatRequest(BaseModel):
    question: str
    history: list["ChatTurn"] = Field(default_factory=list, max_length=20)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4_000)


class ChatResponse(BaseModel):
    answer: str


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("3/minute")
def chat(request: Request, body: ChatRequest, ask_fn=Depends(get_ask_fn)) -> ChatResponse:
    try:
        answer = (
            ask_fn(body.question)
            if not body.history
            else ask_fn(body.question, history=[turn.model_dump() for turn in body.history])
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return ChatResponse(answer=answer)
