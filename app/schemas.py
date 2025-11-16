"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str = Field(..., description="Natural language question about member data")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "When is Layla planning her trip to London?"
            }
        }


class AnswerResponse(BaseModel):
    """Response model for answers."""
    answer: str = Field(..., description="Answer to the question based on member data")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Layla Kawaguchi is planning a trip to London, but the exact date is not specified in the messages."
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status of the service")


class CacheClearResponse(BaseModel):
    """Response model for cache clear operation."""
    status: str = Field(..., description="Status of the cache clear operation")
    message: str = Field(..., description="Detailed message about the operation")

