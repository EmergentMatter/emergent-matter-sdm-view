"""Read an `.sdm` part file into typed dataclasses.

Source-of-truth format is defined by `emergent-matter-sdm-core`'s versioned
schemas. This module mirrors only the fields needed to drive a Blender
auto-panel: name, params (with bounds + units), and stub pointers to
materials/couplings/objectives/constraints for later expansion.

**Why this exists when sdm-core already ships `load()`.** sdm-core's loader
returns a full `Part`, and importing it pulls in jax. This package's `io`
layer is the one place that must stay importable from any Python, including
a headless `.vdb` writer with no solver installed, so it reads the JSON
directly. The cost is that this file has to track the wire format by hand.

Version policy follows sdm-core's, deliberately: `schema_version` is the
MINIMUM a reader must support, not an exact match to demand. A document
declaring a version this loader has not seen is read on a best-effort basis
with a warning, because the fields read here have been stable across every
version so far. A malformed document still raises.

Pure Python: no bpy. Safe to import outside Blender.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# Mirrors software_defined_matter.model.KNOWN_SCHEMA_VERSIONS. Extend it once
# a new version has been read successfully, not on faith.
SDM_SCHEMA_VERSIONS = ("0.1", "0.2", "0.3", "0.4")

# What a document that declares nothing is assumed to be, matching sdm-core's
# FALLBACK_SCHEMA_VERSION. The oldest version is the safe reading: it is the
# one that promises the least.
SDM_FALLBACK_SCHEMA_VERSION = "0.1"


@dataclass
class SdmParam:
    name: str
    value: float
    free: bool = False
    bounds: tuple[float, float] | None = None
    unit: str = ""


@dataclass
class SdmPart:
    name: str
    params: list[SdmParam] = field(default_factory=list)
    # Stub pointers: kept as raw dicts for now; later we'll mirror the
    # full schema as typed dataclasses (materials, couplings, etc.).
    materials_raw: list[dict] = field(default_factory=list)
    couplings_raw: list[dict] = field(default_factory=list)
    objectives_raw: list[dict] = field(default_factory=list)
    constraints_raw: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def load_sdm_part(path: str | Path) -> SdmPart:
    """Parse an `.sdm` file into an `SdmPart`.

    Warns, and reads anyway, on a `schema_version` this loader has not been
    checked against. Raises `KeyError` or `ValueError` from the parse itself
    if the document is malformed.
    """
    raw = json.loads(Path(path).read_text())
    version = str(raw.get("schema_version", SDM_FALLBACK_SCHEMA_VERSION))
    if version not in SDM_SCHEMA_VERSIONS:
        warnings.warn(
            f".sdm schema_version {version!r} is not one this loader has been "
            f"checked against ({SDM_SCHEMA_VERSIONS}); reading on a "
            f"best-effort basis.",
            stacklevel=2,
        )
    params: list[SdmParam] = []
    for key, p in (raw.get("params") or {}).items():
        bounds = p.get("bounds")
        bounds_t: tuple[float, float] | None = (
            (float(bounds[0]), float(bounds[1])) if bounds and len(bounds) == 2 else None
        )
        params.append(
            SdmParam(
                name=str(p.get("name", key)),
                value=float(p["value"]),
                free=bool(p.get("free", False)),
                bounds=bounds_t,
                unit=str(p.get("unit", "")),
            )
        )
    return SdmPart(
        name=str(raw["name"]),
        params=params,
        materials_raw=list(raw.get("materials") or []),
        couplings_raw=list(raw.get("couplings") or []),
        objectives_raw=list(raw.get("objectives") or []),
        constraints_raw=list(raw.get("constraints") or []),
        metadata=dict(raw.get("metadata") or {}),
    )
