import os
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import db, create_document, get_documents

# Optional OpenAI import (used only if API key is available)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
try:
    if OPENAI_API_KEY:
        from openai import OpenAI  # type: ignore
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        openai_client = None
except Exception:
    openai_client = None

app = FastAPI(title="ChapterSmith AI – Complete Story Builder")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Models
# -------------------------------
POVMode = Literal["female", "male", "dual"]

class CreateProjectRequest(BaseModel):
    name: str
    outline: str
    chapter_count: int = Field(..., ge=3, le=6)
    pov_mode: POVMode = "female"
    default_pov: Literal["female", "male"] = "female"
    rules: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

class ProjectResponse(BaseModel):
    id: str
    name: str
    outline: str
    chapter_count: int
    pov_mode: POVMode
    default_pov: Literal["female", "male"]
    rules: List[str]
    tags: List[str]
    created_at: datetime
    updated_at: datetime

class ChapterResponse(BaseModel):
    id: str
    project_id: str
    number: int
    title: str
    content: str
    words: int
    pov: Literal["female", "male"]
    status: Literal["pending", "draft", "final"]
    meta: Dict[str, Any] = {}

class UpdateChapterRequest(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    status: Optional[Literal["pending", "draft", "final"]] = None
    pov: Optional[Literal["female", "male"]] = None

# -------------------------------
# Helpers
# -------------------------------

def _collection(name: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    return db[name]


def resolve_chapter_pov(pov_mode: POVMode, default_pov: Literal["female", "male"], chapter_number: int) -> Literal["female", "male"]:
    if pov_mode == "dual":
        return "female" if chapter_number % 2 == 1 else "male"
    return default_pov


def build_generation_prompt(project: dict, chapter_number: int, chapter_pov: str) -> str:
    # Core writing rules from the product spec, condensed for the model
    rules_block = "\n".join(project.get("rules") or [])
    outline = project.get("outline", "")
    pov_sentence = (
        f"Use deep first-person POV from the {chapter_pov} lead’s perspective, staying close to their thoughts, emotions, and physical sensations."
    )

    strict_wording = (
        "Each chapter must be strictly between 1400 and 1800 words. Do not write less than 1400 words, and do not exceed 1800 words. Ensure the chapter feels complete and cohesive while staying within this word count."
    )

    universal_rules = f"""
You must follow all of these style instructions:
- Natural, grounded, emotionally authentic voice with smooth pacing and clear internal monologue.
- Avoid metaphors, similes, purple prose, and dramatic fragments. Use plain, human wording.
- Balance action and introspection. Show emotions through thoughts, dialogue, and subtle body cues.
- Dialogue must sound natural and reveal power dynamics and hidden feelings. Do not name emotions directly.
- Maintain continuity with previous chapters. Start with tension or action; end with a hook or strong emotional beat.
- Do not use dash-separated adjective lists or mood-heavy sentence fragments. Prefer full, connected sentences.
- Do not use contractions like "I'd"; prefer full forms like "I had" if that matches the selected tone preference.
- Keep narration cinematic, character-driven, and emotionally immersive.

POV Rules to apply:
- All chapters use deep, immersive perspective.
- Default POV: {project.get('default_pov', 'female').capitalize()} (unless Dual is selected).
- Dual POV alternates automatically between female (odd chapters) and male (even chapters).

{pov_sentence}
{strict_wording}
""".strip()

    return f"""
Write Chapter {chapter_number} of a continuous story based on the following outline. Continue naturally from prior chapters, maintaining tone, pacing, and character continuity.

OUTLINE (foundation to follow strictly):
{outline}

WRITING RULES:
{universal_rules}

ADDITIONAL USER RULES:
{rules_block}

OUTPUT FORMAT:
1) A concise but evocative Chapter Title on the first line.
2) Full chapter text (strictly 1400–1800 words), cohesive and complete, ending with a meaningful hook.
""".strip()


def serialize_project(doc: dict) -> ProjectResponse:
    return ProjectResponse(
        id=str(doc.get("_id")),
        name=doc["name"],
        outline=doc["outline"],
        chapter_count=doc["chapter_count"],
        pov_mode=doc["pov_mode"],
        default_pov=doc["default_pov"],
        rules=doc.get("rules", []),
        tags=doc.get("tags", []),
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
    )


def serialize_chapter(doc: dict) -> ChapterResponse:
    return ChapterResponse(
        id=str(doc.get("_id")),
        project_id=str(doc.get("project_id")),
        number=doc["number"],
        title=doc.get("title", ""),
        content=doc.get("content", ""),
        words=doc.get("words", 0),
        pov=doc.get("pov", "female"),
        status=doc.get("status", "draft"),
        meta=doc.get("meta", {}),
    )

# -------------------------------
# Routes
# -------------------------------
@app.get("/")
def read_root():
    return {"message": "ChapterSmith AI Backend running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["connection_status"] = "Connected"
            try:
                response["collections"] = db.list_collection_names()
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    response["llm_enabled"] = bool(openai_client)
    return response


@app.post("/api/projects", response_model=ProjectResponse)
def create_project(req: CreateProjectRequest):
    projects = _collection("project")
    now = datetime.utcnow()
    data = {
        "name": req.name,
        "outline": req.outline,
        "chapter_count": req.chapter_count,
        "pov_mode": req.pov_mode,
        "default_pov": req.default_pov,
        "rules": req.rules,
        "tags": req.tags,
        "created_at": now,
        "updated_at": now,
    }
    new_id = create_document("project", data)
    inserted = projects.find_one({"_id": projects._Database__client.get_default_database().codec_options.document_class()._id if False else None})
    # fallback fetch by id
    from bson import ObjectId
    inserted = projects.find_one({"_id": ObjectId(new_id)})
    return serialize_project(inserted)


@app.get("/api/projects", response_model=List[ProjectResponse])
def list_projects():
    cursor = _collection("project").find().sort("created_at", -1)
    return [serialize_project(doc) for doc in cursor]


@app.get("/api/projects/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str):
    from bson import ObjectId
    doc = _collection("project").find_one({"_id": ObjectId(project_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found")
    return serialize_project(doc)


@app.post("/api/projects/{project_id}/chapters/init", response_model=List[ChapterResponse])
def init_chapters(project_id: str):
    from bson import ObjectId
    proj = _collection("project").find_one({"_id": ObjectId(project_id)})
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    chapters_col = _collection("chapter")
    existing = list(chapters_col.find({"project_id": project_id}))
    if existing:
        return [serialize_chapter(c) for c in sorted(existing, key=lambda x: x.get("number", 0))]

    docs = []
    for num in range(1, int(proj["chapter_count"]) + 1):
        pov = resolve_chapter_pov(proj["pov_mode"], proj["default_pov"], num)
        doc = {
            "project_id": project_id,
            "number": num,
            "title": f"Chapter {num}",
            "content": "",
            "words": 0,
            "pov": pov,
            "status": "pending",
            "meta": {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        _id = create_document("chapter", doc)
        from bson import ObjectId
        docs.append(chapters_col.find_one({"_id": ObjectId(_id)}))

    return [serialize_chapter(d) for d in docs]


@app.get("/api/projects/{project_id}/chapters", response_model=List[ChapterResponse])
def list_chapters(project_id: str):
    docs = list(_collection("chapter").find({"project_id": project_id}).sort("number", 1))
    return [serialize_chapter(d) for d in docs]


@app.get("/api/projects/{project_id}/chapters/{number}", response_model=ChapterResponse)
def get_chapter(project_id: str, number: int):
    doc = _collection("chapter").find_one({"project_id": project_id, "number": number})
    if not doc:
        raise HTTPException(status_code=404, detail="Chapter not found")
    return serialize_chapter(doc)


@app.patch("/api/projects/{project_id}/chapters/{number}", response_model=ChapterResponse)
def update_chapter(project_id: str, number: int, req: UpdateChapterRequest):
    chapters = _collection("chapter")
    doc = chapters.find_one({"project_id": project_id, "number": number})
    if not doc:
        raise HTTPException(status_code=404, detail="Chapter not found")

    update: Dict[str, Any] = {}
    if req.title is not None:
        update["title"] = req.title
    if req.content is not None:
        update["content"] = req.content
        update["words"] = len((req.content or "").split())
    if req.status is not None:
        update["status"] = req.status
    if req.pov is not None:
        update["pov"] = req.pov

    update["updated_at"] = datetime.utcnow()

    _collection("chapter").update_one({"_id": doc["_id"]}, {"$set": update})
    doc = chapters.find_one({"_id": doc["_id"]})
    return serialize_chapter(doc)


@app.post("/api/projects/{project_id}/chapters/{number}/generate")
def generate_chapter(project_id: str, number: int):
    from bson import ObjectId
    projects = _collection("project")
    chapters = _collection("chapter")

    proj = projects.find_one({"_id": ObjectId(project_id)})
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    ch = chapters.find_one({"project_id": project_id, "number": number})
    if not ch:
        raise HTTPException(status_code=404, detail="Chapter not found")

    pov = ch.get("pov") or resolve_chapter_pov(proj["pov_mode"], proj["default_pov"], number)
    prompt = build_generation_prompt(proj, number, pov)

    if not openai_client:
        # No LLM available: return the prompt so the user can copy it
        return {
            "mode": "prompt_only",
            "message": "No OpenAI API key configured on the server. Use this prompt in your model and paste the result back via Edit Chapter.",
            "prompt": prompt,
        }

    # Call OpenAI for generation with strong word target
    system = "You are a professional fiction ghostwriter. You obey word counts and style constraints precisely."
    target_words = 1650

    try:
        completion = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt + f"\n\nTarget word count: {target_words} words (within 1400-1800)."},
            ],
            temperature=0.7,
        )
        text = completion.choices[0].message.content.strip()
    except Exception as e:
        # Fallback to prompt-only if API fails
        return {
            "mode": "prompt_only",
            "message": f"LLM error: {str(e)[:200]}",
            "prompt": prompt,
        }

    words = len(text.split())
    title = "Chapter " + str(number)
    # Extract first line as title if clearly labeled
    if "\n" in text:
        first_line, rest = text.split("\n", 1)
        if len(first_line) < 120 and first_line.lower().startswith("chapter") or len(first_line.split()) <= 12:
            title = first_line.strip()
            text = rest.strip()

    chapters.update_one({"_id": ch["_id"]}, {"$set": {
        "title": title,
        "content": text,
        "words": words,
        "status": "draft",
        "updated_at": datetime.utcnow(),
        "meta.prompt": prompt,
    }})

    updated = chapters.find_one({"_id": ch["_id"]})
    return serialize_chapter(updated)


# Endpoint to expose current schemas (for admin tooling)
@app.get("/schema")
def get_schema_index():
    return {
        "project": {
            "fields": ["name", "outline", "chapter_count", "pov_mode", "default_pov", "rules", "tags"],
        },
        "chapter": {
            "fields": ["project_id", "number", "title", "content", "words", "pov", "status"],
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
