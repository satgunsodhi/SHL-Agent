"""
Pydantic models for the SHL Assessment Recommender API.
Strict schema compliance with the assignment specification.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class ChatMessage(BaseModel):
    """A single message in the conversation history."""
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="The message content")


class ChatRequest(BaseModel):
    """
    POST /chat request body.
    Stateless — every call carries the full conversation history.
    """
    messages: List[ChatMessage] = Field(
        ...,
        description="Full conversation history as an ordered list of messages",
        min_length=1
    )


class Recommendation(BaseModel):
    """A single assessment recommendation from the SHL catalog."""
    name: str = Field(..., description="Assessment name from catalog")
    url: str = Field(..., description="Canonical SHL catalog URL")
    test_type: str = Field(..., description="Test type code(s), e.g. 'K', 'P', 'A, P'")


class ChatResponse(BaseModel):
    """
    POST /chat response body.
    - recommendations is null when gathering context or refusing.
    - recommendations is an array of 1-10 items when the agent has committed to a shortlist.
    - end_of_conversation is true only when the agent considers the task complete.
    """
    reply: str = Field(..., description="The agent's natural language reply")
    recommendations: Optional[List[Recommendation]] = Field(
        None,
        description="Null when clarifying/refusing; 1-10 items when recommending"
    )
    end_of_conversation: bool = Field(
        False,
        description="True only when the agent considers the conversation complete"
    )


class HealthResponse(BaseModel):
    """GET /health response body."""
    status: str = "ok"
