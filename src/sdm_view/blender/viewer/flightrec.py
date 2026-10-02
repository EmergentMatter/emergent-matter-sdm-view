"""Always-on flight recorder: one JSON line per notable viewer event.

Exists because the failure mode under study (a viewport draw that takes
seconds, back-to-back, starving WindowServer until terminal windows blank)
kills the very session that was watching for it. Evidence has to land on
disk the moment it happens, not in a terminal that may be dead by then.

Events land in ``$SDM_FLIGHT_LOG`` (default ``/tmp/sdm_flight.jsonl``), one
JSON object per line: ``t`` wall-clock ISO time, ``mono`` monotonic seconds,
``ev`` event name, plus whatever fields the call site passes. Recording is
best-effort and must never break a draw: every failure is swallowed.

Cost when nothing is wrong is one ``perf_counter`` pair per draw: the
draw handler only calls :func:`log` for draws over its slow threshold, and
every other instrumented event (shader create, PSO warm-up, rebuild stages)
is rare by construction.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

_PATH = os.environ.get("SDM_FLIGHT_LOG", "/tmp/sdm_flight.jsonl")


def log(ev: str, **fields) -> None:
    try:
        rec = {
            "t": datetime.now().isoformat(timespec="milliseconds"),
            "mono": round(time.monotonic(), 3),
            "ev": ev,
        }
        rec.update(fields)
        with open(_PATH, "a") as f:  # noqa: PTH123  # os.path usage predates the pathlib convention; not touched in this mechanical pass
            f.write(json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001  # the recorder must never take down a draw
        pass
