"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- Project -> "project" collection
- Chapter -> "chapter" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any
from datetime import datetime

# Core domain schemas for ChapterSmith AI

POVMode = Literal["female", "male", "dual"]

class Chapter(BaseModel):
    """
    Chapters collection schema
    Collection name: "chapter"
    """
    project_id: str = Field(..., description="Related project id as string")
    number: int = Field(..., ge=1, description="Chapter number starting at 1")
    title: str = Field("", description="Chapter title")
    content: str = Field("", description="Full chapter text")
    words: int = Field(0, ge=0, description="Word count of content")
    pov: Literal["female", "male"] = Field("female", description="Resolved POV for this chapter")
    status: Literal["pending", "draft", "final"] = Field("draft", description="Generation status")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata like prompts or notes")

class Project(BaseModel):
    """
    Projects collection schema
    Collection name: "project"
    """
    name: str = Field(..., description="Project name")
    outline: str = Field(..., description="User provided outline text")
    chapter_count: int = Field(..., ge=3, le=6, description="Number of chapters 3-6")
    pov_mode: POVMode = Field("female", description="female | male | dual")
    default_pov: Literal["female", "male"] = Field("female", description="Default POV when not dual")
    rules: List[str] = Field(default_factory=list, description="Writing rules applied")
    tags: List[str] = Field(default_factory=list, description="Optional tags/genres")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

# Note: The Flames database viewer will automatically read these from GET /schema
