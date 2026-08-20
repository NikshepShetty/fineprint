from fastapi import FastAPI

from fineprint.api.routers import chat, documents, predict

app = FastAPI()

app.include_router(predict.router)
app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/health")
def health():
    return {"status": "ok"}
