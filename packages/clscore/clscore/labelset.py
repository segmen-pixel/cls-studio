# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Cls-Studio Contributors
"""The operator's judgement about a feature store, kept apart from the features.

A label set is a pure-metadata document: for each image in a
:class:`clscore.store.FeatureStore` it records which tier the operator put it
in, under which defect label, how severe it is, and which patches of the
image's grid are the defect. Nothing here costs a forward pass, so a label
set can be rewritten as often as the operator changes their mind — which is
the point, because *the assignment is what the model learns*.

Several label sets can coexist over one store. That makes the comparison
that used to be impossible cheap: "what if these borderline images counted as
NG?" is a second label set and a re-assemble, not a re-teach.

Marks address the image's **patch grid**, never its stored rows. The two
differ as soon as a per-image coreset cap drops patches, and only the grid
index is stable across a re-ingest or a change of cap — storing row positions
here would silently move every mark the first time an image was re-ingested.

Disk layout, relative to the label-sets directory::

    <labelset_id>.json
    .active                 id of the label set the bank is currently built from
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .bank import DEFAULT_LABEL, Tier, safe_label
from .fsio import atomic_write_text
from .incident import DEFAULT_SEVERITY, SEVERITY_HEAVY, SEVERITY_LIGHT

__all__ = [
    "ACTIVE_MARKER",
    "LABELSETS_SUBDIR",
    "Assignment",
    "LabelSet",
    "list_labelsets",
    "read_active_id",
    "slug_labelset_id",
    "write_active_id",
]

LABELSETS_SUBDIR = "labelsets"
ACTIVE_MARKER = ".active"
DEFAULT_LABELSET_ID = "standard"
DEFAULT_LABELSET_NAME = "標準"

_SLUG_RE = re.compile(r"[^a-z0-9-]+")


def slug_labelset_id(name: str) -> str:
    """Filesystem-safe label-set id from a display name."""
    s = _SLUG_RE.sub("-", str(name).strip().lower()).strip("-")
    return s[:64] or DEFAULT_LABELSET_ID


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Assignment:
    """What the operator decided about one stored image."""

    tier: Tier = "normal"
    # Defect class, only meaningful for the labelled tiers. Stored already
    # sanitised so the assembled bank's on-disk filenames are predictable.
    label: str = ""
    # Base severity for every row this image contributes. Marks override it.
    severity: int = DEFAULT_SEVERITY
    # Grid patch indices the operator marked as the defect itself.
    marks: list[int] = field(default_factory=list)
    # Source rectangles, normalised to the image, kept purely so the UI can
    # redraw and edit them. ``marks`` is the scoring-facing artifact.
    rects: list[dict] = field(default_factory=list)

    def resolved_label(self) -> str:
        """On-disk label for this assignment (``""`` for the normal tier)."""
        if self.tier == "normal":
            return ""
        # ``_default`` must survive verbatim: safe_label strips leading
        # underscores and would rewrite it into a different bucket.
        return self.label if self.label == DEFAULT_LABEL else safe_label(self.label)

    def clamped_severity(self) -> int:
        return max(SEVERITY_LIGHT, min(SEVERITY_HEAVY, int(self.severity)))


@dataclass
class LabelSet:
    """A named set of assignments over one feature store."""

    id: str = DEFAULT_LABELSET_ID
    name: str = DEFAULT_LABELSET_NAME
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    # store entry id -> Assignment. An entry absent from this dict is
    # unassigned: it stays in the store and contributes nothing to the bank,
    # which is how "not labelled yet" is represented.
    assignments: dict[str, Assignment] = field(default_factory=dict)

    # ---- mutation ---------------------------------------------------------

    def assign(
        self,
        entry_id: str,
        tier: Tier,
        label: str = "",
        severity: int = DEFAULT_SEVERITY,
    ) -> Assignment:
        """Put one image in a tier, preserving its marks when they still apply.

        Marks address the patch grid, so they stay meaningful when an image
        moves between the labelled tiers — the operator circled the same
        pixels either way. Moving to ``normal`` drops them: a nominal image
        has no defect regions by definition, and keeping them would silently
        resurrect the marks if it were moved back.
        """
        prev = self.assignments.get(entry_id)
        a = Assignment(tier=tier, label=label, severity=severity)
        if prev is not None and tier != "normal":
            a.marks = list(prev.marks)
            a.rects = list(prev.rects)
        self.assignments[entry_id] = a
        return a

    def unassign(self, entry_id: str) -> bool:
        """Return the image to the unlabelled pool. True when it was assigned."""
        return self.assignments.pop(entry_id, None) is not None

    def mark(self, entry_id: str, marks: list[int], rects: list[dict] | None = None) -> Assignment:
        """Replace one image's defect marks.

        Replace, not merge: re-marking has to be able to remove a region, and
        a merge would make every correction additive.
        """
        a = self.assignments.get(entry_id)
        if a is None:
            raise KeyError(f"image {entry_id!r} is not assigned to a tier yet")
        a.marks = sorted({int(v) for v in marks})
        a.rects = list(rects or [])
        return a

    # ---- queries ----------------------------------------------------------

    def tier_of(self, entry_id: str) -> str:
        a = self.assignments.get(entry_id)
        return a.tier if a is not None else ""

    def counts(self) -> dict[str, int]:
        out = {"normal": 0, "critical": 0, "negative": 0}
        for a in self.assignments.values():
            if a.tier in out:
                out[a.tier] += 1
        return out

    def labels(self, tier: Tier) -> list[str]:
        return sorted(
            {a.resolved_label() for a in self.assignments.values() if a.tier == tier}
        )

    # ---- persistence ------------------------------------------------------

    def to_json(self) -> str:
        payload = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at or _now(),
            "updated_at": self.updated_at or _now(),
            "assignments": {k: asdict(v) for k, v in sorted(self.assignments.items())},
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.updated_at = _now()
        if not self.created_at:
            self.created_at = self.updated_at
        path = directory / f"{self.id}.json"
        atomic_write_text(path, self.to_json())
        return path

    @classmethod
    def load(cls, path: Path) -> LabelSet:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assignments = {
            k: Assignment(
                **{f: v[f] for f in v if f in Assignment.__dataclass_fields__}
            )
            for k, v in (data.get("assignments") or {}).items()
        }
        return cls(
            id=str(data.get("id") or Path(path).stem),
            name=str(data.get("name") or Path(path).stem),
            description=str(data.get("description") or ""),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
            assignments=assignments,
        )


def list_labelsets(directory: Path) -> list[LabelSet]:
    """Every label set in ``directory``, oldest id first. Unreadable ones are skipped."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    out: list[LabelSet] = []
    for p in sorted(directory.glob("*.json")):
        try:
            out.append(LabelSet.load(p))
        except (OSError, ValueError):
            continue  # a torn write must not make the whole list unreadable
    return out


def read_active_id(directory: Path) -> str | None:
    try:
        return (Path(directory) / ACTIVE_MARKER).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def write_active_id(directory: Path, labelset_id: str) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_text(directory / ACTIVE_MARKER, labelset_id)
