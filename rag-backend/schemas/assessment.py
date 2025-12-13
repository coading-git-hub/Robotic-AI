from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import List, Optional


class Answer(BaseModel):
    question_id: UUID
    user_answer: str
    correct_answer: Optional[str] = None
    is_correct: Optional[bool] = None


class AssessmentRequest(BaseModel):
    assessment_id: UUID
    module: Optional[str] = None
    lesson: Optional[str] = None
    answers: List[Answer]
    time_taken: Optional[int] = None  # Time in seconds


class AssessmentResponse(BaseModel):
    assessment_id: UUID
    user_id: UUID
    score: float
    message: str


class AssessmentResult(BaseModel):
    assessment_id: UUID
    user_id: UUID
    score: float
    total_questions: int
    correct_answers: int
    time_taken: Optional[int] = None
    answers: List[Answer]


class GradebookEntry(BaseModel):
    assessment_id: UUID
    assessment_name: str
    module: str
    score: float
    max_score: float = 100.0
    date_completed: str