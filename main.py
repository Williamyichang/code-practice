import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field


# ============================================================
# Environment variables
# ============================================================

SKILL_CREATOR_API_KEY = os.getenv("SKILL_CREATOR_API_KEY", "")

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")

if not AZURE_OPENAI_API_KEY:
    raise RuntimeError("Missing AZURE_OPENAI_API_KEY")

if not AZURE_OPENAI_ENDPOINT:
    raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT")

if not AZURE_OPENAI_DEPLOYMENT:
    raise RuntimeError("Missing AZURE_OPENAI_DEPLOYMENT")


# Azure OpenAI v1 endpoint format:
# https://<resource-name>.openai.azure.com/openai/v1/
client = OpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1/",
)


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="Dynamic Skill Creator API",
    version="0.1.0",
    description=(
        "A FastAPI service for creating reusable skill documents from "
        "Langflow conversation summaries."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production 建議改成你的 Langflow / frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Security
# ============================================================

def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if not SKILL_CREATOR_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server SKILL_CREATOR_API_KEY is not configured.",
        )

    if x_api_key != SKILL_CREATOR_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid X-API-Key.",
        )


# ============================================================
# Request / response models
# ============================================================

class SkillCreateRequest(BaseModel):
    conversation_id: Optional[str] = Field(
        default=None,
        description="Langflow session id, conversation id, or trace id.",
    )

    user_goal: Optional[str] = Field(
        default=None,
        description="The user's final goal or task purpose.",
    )

    conversation_text: Optional[str] = Field(
        default=None,
        description="Full Q&A content or selected important turns.",
    )

    agent_summary: Optional[str] = Field(
        default=None,
        description="Summary Agent output generated from the whole Q&A process.",
    )

    final_answer: Optional[str] = Field(
        default=None,
        description="Final answer produced before the user asks to create a skill.",
    )

    source_system: str = Field(
        default="Langflow",
        description="Source system name.",
    )

    language: Literal["zh-TW", "en"] = Field(
        default="zh-TW",
        description="Skill output language.",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata from Langflow, flow id, user id, tags, etc.",
    )


class SkillDocument(BaseModel):
    skill_name: str = Field(
        description="Reusable skill name. Avoid one-time names."
    )

    scenario_description: str = Field(
        description="When this skill should be used."
    )

    trigger_conditions: List[str] = Field(
        description="Conditions that should trigger this skill."
    )

    anti_trigger_conditions: List[str] = Field(
        description="Conditions where this skill should not be used."
    )

    prompt_fragment: str = Field(
        description="Reusable behavior instruction that can be injected into an Agent prompt."
    )

    workflow_orchestration_guide: List[str] = Field(
        description=(
            "Workflow guide derived from the whole Q&A process. "
            "This must not be a fixed generic workflow."
        )
    )

    input_requirements: List[str] = Field(
        description="Required user inputs, data, documents, or context."
    )

    tool_requirements: List[str] = Field(
        description="Required tools, APIs, databases, or files."
    )

    output_contract: str = Field(
        description="Expected output format and structure."
    )

    validation_checklist: List[str] = Field(
        description="Checks before the skill output is accepted."
    )

    risk_notes: List[str] = Field(
        description="Known limitations, risks, and failure modes."
    )

    reusable_pattern: str = Field(
        description="The general reusable task pattern extracted from the conversation."
    )

    non_reusable_details_to_avoid: List[str] = Field(
        description="One-time details that should not be hard-coded into the skill."
    )

    status: Literal["draft", "validated", "needs_review"] = Field(
        description="Skill maturity status."
    )


class SkillValidationResponse(BaseModel):
    is_valid: bool
    score: int
    warnings: List[str]
    suggestions: List[str]


class SkillCreateResponse(BaseModel):
    skill_id: str
    created_at: str
    source_conversation_id: Optional[str]
    skill_json: SkillDocument
    skill_markdown: str
    validation: SkillValidationResponse


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT_ZH_TW = """
你是一個 Skill Creator Agent。

你的任務是根據 Langflow 傳入的完整問答內容、Summary Agent 摘要、最終回答與使用者目標，
產生一份可重用的 Skill 文件。

你不是在摘要聊天紀錄，而是在萃取「可重複使用的任務能力」。

重要規則：

1. Skill 必須是可重用能力，不是單次任務紀錄。
2. workflow_orchestration_guide 必須根據整個問答過程萃取產生。
3. workflow_orchestration_guide 不能是固定 Step 1 / Step 2 / Step 3 通用模板。
4. 必須分析：
   - 使用者最初提出的問題
   - 使用者後續補充的條件
   - Agent 的分析方法
   - 使用過的工具或資料來源
   - 中間修正方向
   - 最後被確認的輸出偏好
5. 不要把單次任務中的公司名稱、日期、價格、檔案名稱、短期事件寫死成永久規則。
6. 如果單次資訊可以作為例子，可以保留為 example，但不能變成硬性規則。
7. 如果資訊不足以形成完整 skill，status 必須設為 draft 或 needs_review。
8. 必須清楚區分：
   - 可重用任務模式
   - 一次性細節
   - 觸發條件
   - 不應觸發條件
   - 工具需求
   - 輸出契約
   - 驗證標準
9. 請使用繁體中文輸出。
"""


def build_user_prompt(payload: SkillCreateRequest) -> str:
    return f"""
請根據以下 Langflow 問答內容產生一份可重用 Skill 文件。

# User Goal
{payload.user_goal or "未提供"}

# Conversation Text
{payload.conversation_text or "未提供"}

# Agent Summary
{payload.agent_summary or "未提供"}

# Final Answer
{payload.final_answer or "未提供"}

# Source System
{payload.source_system}

# Metadata
{payload.metadata}

請特別注意：
- 工作流程編排指引必須從整個問答內容萃取。
- 不要輸出固定的通用流程模板。
- 應該描述這次問答中「實際成功完成任務的方法」。
- 若 Summary Agent 已經提供可重用任務模式，請優先使用 Summary Agent 的內容。
"""


# ============================================================
# Markdown renderer
# ============================================================

def _bullet(items: List[str]) -> str:
    if not items:
        return "- 無"
    return "\n".join(f"- {item}" for item in items)


def render_skill_markdown(skill: SkillDocument) -> str:
    return f"""# {skill.skill_name}

## 1. 使用場景描述

{skill.scenario_description}

---

## 2. 觸發條件

{_bullet(skill.trigger_conditions)}

---

## 3. 不使用條件

{_bullet(skill.anti_trigger_conditions)}

---

## 4. 行為描述 Prompt Fragment

{skill.prompt_fragment}

---

## 5. 工作流程編排指引

> 本工作流程必須依據整個問答內容萃取產生，而不是固定模板。

{_bullet(skill.workflow_orchestration_guide)}

---

## 6. 輸入資料需求

{_bullet(skill.input_requirements)}

---

## 7. 工具需求

{_bullet(skill.tool_requirements)}

---

## 8. 輸出格式契約

{skill.output_contract}

---

## 9. 驗證標準

{_bullet(skill.validation_checklist)}

---

## 10. 風險與限制

{_bullet(skill.risk_notes)}

---

## 11. 可重用任務模式

{skill.reusable_pattern}

---

## 12. 不應寫死的一次性細節

{_bullet(skill.non_reusable_details_to_avoid)}

---

## 13. 狀態

{skill.status}
"""


# ============================================================
# Validator
# ============================================================

def validate_skill(skill: SkillDocument) -> SkillValidationResponse:
    warnings: List[str] = []
    suggestions: List[str] = []
    score = 100

    if len(skill.skill_name.strip()) < 4:
        warnings.append("技能名稱過短，可能不夠具體。")
        suggestions.append("使用能代表可重用能力的名稱，不要只用單次任務名稱。")
        score -= 10

    if len(skill.scenario_description.strip()) < 20:
        warnings.append("使用場景描述可能太短。")
        suggestions.append("補充使用者通常如何提出此類問題，以及此 Skill 解決什麼任務。")
        score -= 10

    if not skill.trigger_conditions:
        warnings.append("缺少觸發條件，未來 Skill Router 可能難以判斷何時使用。")
        suggestions.append("加入 3 到 5 個明確觸發條件。")
        score -= 10

    if not skill.anti_trigger_conditions:
        warnings.append("缺少不使用條件，可能造成 Skill 過度觸發。")
        suggestions.append("加入此 Skill 不應使用的情況。")
        score -= 10

    if len(skill.workflow_orchestration_guide) < 3:
        warnings.append("工作流程編排指引太短，可能不足以重用。")
        suggestions.append("根據完整問答內容補充實際任務流程。")
        score -= 15

    workflow_text = "\n".join(skill.workflow_orchestration_guide)

    generic_phrases = [
        "Step 1",
        "Step 2",
        "Step 3",
        "理解使用者問題",
        "查詢資料",
        "輸出結果",
        "固定模板",
    ]

    matched_generic = [phrase for phrase in generic_phrases if phrase in workflow_text]

    if len(matched_generic) >= 3:
        warnings.append("工作流程可能仍偏向固定通用模板，而非根據問答內容萃取。")
        suggestions.append("改寫 workflow，使其反映本次問答中實際形成的方法、限制與修正方向。")
        score -= 15

    if not skill.prompt_fragment.strip():
        warnings.append("缺少 Prompt Fragment。")
        suggestions.append("加入可直接注入 Agent system prompt 的行為描述。")
        score -= 15

    if not skill.validation_checklist:
        warnings.append("缺少驗證標準。")
        suggestions.append("加入輸出前應檢查的條件。")
        score -= 10

    score = max(0, min(100, score))
    is_valid = score >= 80 and len(warnings) <= 2 and skill.status != "draft"

    return SkillValidationResponse(
        is_valid=is_valid,
        score=score,
        warnings=warnings,
        suggestions=suggestions,
    )


# ============================================================
# Routes
# ============================================================

@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "dynamic-skill-creator-api",
        "version": "0.1.0",
    }


@app.post(
    "/api/v1/skills/create",
    response_model=SkillCreateResponse,
    dependencies=[Depends(verify_api_key)],
)
def create_skill(payload: SkillCreateRequest) -> SkillCreateResponse:
    if not payload.conversation_text and not payload.agent_summary:
        raise HTTPException(
            status_code=400,
            detail="Either conversation_text or agent_summary is required.",
        )

    try:
        completion = client.beta.chat.completions.parse(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_ZH_TW,
                },
                {
                    "role": "user",
                    "content": build_user_prompt(payload),
                },
            ],
            response_format=SkillDocument,
            temperature=0.2,
        )

        parsed_skill = completion.choices[0].message.parsed

        if parsed_skill is None:
            raise HTTPException(
                status_code=502,
                detail="Azure OpenAI returned no parsed skill document.",
            )

        validation = validate_skill(parsed_skill)
        markdown = render_skill_markdown(parsed_skill)

        return SkillCreateResponse(
            skill_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            source_conversation_id=payload.conversation_id,
            skill_json=parsed_skill,
            skill_markdown=markdown,
            validation=validation,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create skill: {str(exc)}",
        )


@app.post(
    "/api/v1/skills/validate",
    response_model=SkillValidationResponse,
    dependencies=[Depends(verify_api_key)],
)
def validate_skill_endpoint(skill: SkillDocument) -> SkillValidationResponse:
    return validate_skill(skill)


@app.post(
    "/api/v1/skills/render",
    dependencies=[Depends(verify_api_key)],
)
def render_skill_endpoint(skill: SkillDocument) -> Dict[str, str]:
    return {
        "skill_markdown": render_skill_markdown(skill)
    }
