from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: str
    title: str
    chunk_text: str
    similarity: float = Field(ge=0.0, le=1.0)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    include_citations: bool = True


class IngestRequest(BaseModel):
    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation]
    model: str
    tokens_used: int


class HealthResponse(BaseModel):
    status: str
    db: str
