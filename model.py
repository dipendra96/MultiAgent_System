from typing import List, Literal
from pydantic import BaseModel, Field


class KnownFacts(BaseModel):
    goals: List[str] = Field(default_factory=list)
    risk_tolerance: str = ""
    constraints: List[str] = Field(default_factory=list)
    portfolio: str = ""
    time_horizon: str = ""
    liquidity_needs: str = ""
    other_notes: str = ""


class AdvisorIntakeResult(BaseModel):
    known_facts: KnownFacts
    enough_info: bool
    advisor_message: str


class AdvisorReviewResult(BaseModel):
    decision: Literal["ready", "revise"]
    notes: str


class ClientAnswerResult(BaseModel):
    client_message: str


class ClientFeedbackResult(BaseModel):
    client_message: str
    satisfied: bool