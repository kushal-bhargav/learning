from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .service import AgencyConsoleService


class SessionCreateRequest(BaseModel):
    persona_id: str = Field(default="long-distance-partners")
    occasion_id: str | None = None
    budget_hint: str | None = None
    agency_slider: float | None = Field(default=None, ge=0, le=1)
    seed: int = 2026


class StageProposeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overrides: dict[str, Any] = Field(default_factory=dict)


class EditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    human_edit: dict[str, Any] = Field(min_length=1)


class RegenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overrides: dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: float = Field(ge=1, le=5)
    authorship: str | None = None
    open_text: str | None = None
    measures: dict[str, Any] = Field(default_factory=dict)


def create_app(service: AgencyConsoleService | None = None) -> FastAPI:
    app = FastAPI(title="Gift Creator Agency Console API", version="0.1.0")
    console_service = service or AgencyConsoleService()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_service() -> AgencyConsoleService:
        return console_service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/personas")
    def list_personas(api: AgencyConsoleService = Depends(get_service)) -> list[dict[str, Any]]:
        return api.list_personas()

    @app.post("/sessions")
    def create_session(
        request: SessionCreateRequest,
        api: AgencyConsoleService = Depends(get_service),
    ) -> JSONResponse:
        try:
            session = api.create_session(**request.model_dump())
        except (KeyError, StopIteration) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _session_response(api, session.session_id)

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str, api: AgencyConsoleService = Depends(get_service)) -> JSONResponse:
        _require_session(api, session_id)
        return _session_response(api, session_id)

    @app.post("/sessions/{session_id}/stages/{stage}/propose")
    def propose_stage(
        session_id: str,
        stage: str,
        request: StageProposeRequest | None = None,
        api: AgencyConsoleService = Depends(get_service),
    ) -> JSONResponse:
        try:
            api.propose(session_id, stage, (request or StageProposeRequest()).overrides)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _session_response(api, session_id)

    @app.post("/sessions/{session_id}/stages/{stage}/accept")
    def accept_stage(session_id: str, stage: str, api: AgencyConsoleService = Depends(get_service)) -> JSONResponse:
        _require_pending_stage(api, session_id, stage)
        try:
            api.accept(session_id)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _session_response(api, session_id)

    @app.post("/sessions/{session_id}/stages/{stage}/edit")
    def edit_stage(
        session_id: str,
        stage: str,
        request: EditRequest,
        api: AgencyConsoleService = Depends(get_service),
    ) -> JSONResponse:
        _require_pending_stage(api, session_id, stage)
        try:
            api.edit(session_id, request.human_edit)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _session_response(api, session_id)

    @app.post("/sessions/{session_id}/stages/{stage}/regenerate")
    def regenerate_stage(
        session_id: str,
        stage: str,
        request: RegenerateRequest | None = None,
        api: AgencyConsoleService = Depends(get_service),
    ) -> JSONResponse:
        _require_pending_stage(api, session_id, stage)
        try:
            api.regenerate(session_id, stage=stage, overrides=(request or RegenerateRequest()).overrides)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _session_response(api, session_id)

    @app.post("/sessions/{session_id}/stages/{stage}/delegate")
    def delegate_from_stage(session_id: str, stage: str, api: AgencyConsoleService = Depends(get_service)) -> JSONResponse:
        _require_pending_stage(api, session_id, stage)
        try:
            api.delegate(session_id)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _session_response(api, session_id)

    @app.get("/sessions/{session_id}/ledger")
    def ledger(session_id: str, api: AgencyConsoleService = Depends(get_service)) -> dict[str, Any]:
        try:
            return api.ledger(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


    @app.get("/artifacts/{artifact_path:path}")
    def artifact(artifact_path: str, api: AgencyConsoleService = Depends(get_service)) -> FileResponse:
        path = Path(artifact_path)
        if path.is_absolute() or ".." in path.parts:
            raise HTTPException(status_code=400, detail="artifact_path must be a safe relative path")
        resolved = path.resolve()
        allowed = api.generated_dir.resolve()
        try:
            resolved.relative_to(allowed)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="artifact is outside the generated artifact directory") from exc
        if not resolved.exists() or not resolved.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(resolved)
    @app.post("/sessions/{session_id}/feedback")
    def submit_feedback(
        session_id: str,
        request: FeedbackRequest,
        api: AgencyConsoleService = Depends(get_service),
    ) -> dict[str, Any]:
        try:
            return api.submit_feedback(session_id, **request.model_dump())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def _session_response(api: AgencyConsoleService, session_id: str) -> JSONResponse:
    session = api.get_session(session_id)
    payload = json.loads(session.model_dump_json())
    payload["next_stage"] = api.next_stage(session_id)
    payload["ledger"] = api.ledger(session_id)
    return JSONResponse(payload)


def _require_session(api: AgencyConsoleService, session_id: str) -> None:
    try:
        api.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _require_pending_stage(api: AgencyConsoleService, session_id: str, stage: str) -> None:
    session = api.get_session(session_id)
    if not session.stage_log or session.stage_log[-1].status != "pending":
        raise HTTPException(status_code=409, detail="There is no pending agent proposal")
    if session.stage_log[-1].stage != stage:
        raise HTTPException(status_code=409, detail="Action does not match the pending stage")


app = create_app()


