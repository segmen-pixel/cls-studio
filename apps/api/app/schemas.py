# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cls-Studio Contributors
from __future__ import annotations

import json
from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1)
    description: str | None = None
    memo: str | None = None
    tags: list[str] = Field(default_factory=list)


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    memo: str | None = None
    tags: list[str] | None = None


class ProjectRead(BaseModel):
    id: str
    name: str
    description: str | None = None
    memo: str | None = None
    sort_order: int = 0
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True,
    }

    @field_validator("tags", mode="before")
    @classmethod
    def _decode_tags(cls, v):
        if isinstance(v, str):
            try:
                parsed = json.loads(v) if v else []
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        if v is None:
            return []
        return v


class ClassItem(BaseModel):
    id: int
    name: str
    color: list[int]
    active: bool


class ClassesPayload(BaseModel):
    version: int
    ignore_index: int
    classes: list[ClassItem]
