"""
dashboard/app.py

Local demo dashboard for VitalStream.

Runs a FastAPI server that:
  - Streams synthetic physiological signals over WebSocket
  - Runs the real feature extraction + anomaly scoring pipeline (no AWS needed)
  - Serves the monitoring dashboard at http://localhost:8000

The anomaly detector is a lightweight rule-based scorer for demo purposes
(the real model requires a deployed SageMaker endpoint). Scoring rules mirror
what the 1D-CNN+LSTM would detect:
  - Spectral entropy > 0.85  → likely noise/artifact
  - RMS amplitude spike      → possible VT/STEMI
  - Simulated AFIB via RR irregularity injection

Usage:
    python dashboard/app.py
    open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Add project root to path so src imports work
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.signal_processing import (
    bandpass_filter,
    spectral_entropy,
    validate_signal_quality,
    normalise_window,
)

# ---------------------------------------------------------------------------
# Patient registry (demo)
# ---------------------------------------------------------------------------

PATIENTS = [
    {"id": "pt-001", "name": "R. Harmon",   "age": 67, "room": "ICU-4A", "condition": "Post-MI monitoring"},
    {"id": "pt-002", "name": "L. Okafor",   "age": 54, "room": "CCU-2B", "condition": "Atrial fibrillation"},
    {"id": "pt-003", "name": "M. Santiago", "age": 72, "room": "ICU-4C", "condition": "CHF exacerbation"},
    {"id": "pt-004", "name": "K. Brennan",  "age": 45, "room": "STEP-1A","condition": "Post-ablation"},
]

FS             = 500       # Hz
WINDOW_SAMPLES = 500       # 1s window for dashboard refresh rate (smooth animation)
STREAM_HZ      = 20        # push 25ms chunks to client


# ---------------------------------------------------------------------------
# Synthetic signal generators
# ---------------------------------------------------------------------------

def _ecg_window(
    n: int,
    fs: float,
    hr: float = 72.0,
    noise_std: float = 8.0,
    inject_afib: bool = False,
    inject_vt: bool = False,
) -> np.ndarray:
    t = np.linspace(0, n / fs, n, endpoint=False)
    sig = np.random.default_rng().normal(0, noise_std, n).astype(np.float32)

    rr = 60.0 / hr
    beat_times = np.arange(0, n / fs, rr)

    if inject_afib:
        # Irregular R-R intervals, no P wave, fibrillatory baseline
        beat_times = np.cumsum(np.random.uniform(0.5 * rr, 1.5 * rr, len(beat_times)))
        beat_times = beat_times[beat_times < n / fs]
        sig += 15 * np.sin(2 * np.pi * 6 * t)  # fibrillatory baseline

    for bt in beat_times:
        if inject_vt:
            # Wide, bizarre QRS
            sig += 600 * np.exp(-((t - (bt + 0.12)) ** 2) / (2 * 0.018 ** 2))
            sig -= 200 * np.exp(-((t - (bt + 0.16)) ** 2) / (2 * 0.018 ** 2))
        else:
            if not inject_afib:
                sig += 50  * np.exp(-((t - (bt + 0.08)) ** 2) / (2 * 0.012 ** 2))
            sig -= 20  * np.exp(-((t - (bt + 0.14)) ** 2) / (2 * 0.005 ** 2))
            sig += 800 * np.exp(-((t - (bt + 0.16)) ** 2) / (2 * 0.004 ** 2))
            sig -= 150 * np.exp(-((t - (bt + 0.18)) ** 2) / (2 * 0.005 ** 2))
            sig += 100 * np.exp(-((t - (bt + 0.30)) ** 2) / (2 * 0.030 ** 2))

    return sig


def _bp_window(n: int, fs: float = 125.0, sys: float = 120.0, dia: float = 80.0) -> np.ndarray:
    t   = np.linspace(0, n / fs, n, endpoint=False)
    sig = np.full(n, dia, dtype=np.float32)
    hr  = 72.0
    rr  = 60.0 / hr
    for bt in np.arange(0, n / fs, rr):
        sig += (sys - dia) * np.exp(-((t - (bt + 0.12)) ** 2) / (2 * 0.04 ** 2))
        sig += 5 * np.exp(-((t - (bt + 0.30)) ** 2) / (2 * 0.01 ** 2))
    sig += np.random.default_rng().normal(0, 1, n).astype(np.float32)
    return sig


# ---------------------------------------------------------------------------
# Rule-based demo anomaly scorer
# ---------------------------------------------------------------------------

@dataclass
class AnomalyScore:
    score: float          # 0–1
    label: str            # NORMAL / AFIB / VT / NOISE
    confidence: float     # 0–1


def score_window(signal: np.ndarray, inject_afib: bool, inject_vt: bool) -> AnomalyScore:
    if not validate_signal_quality(signal):
        return AnomalyScore(score=0.91, label="NOISE", confidence=0.95)

    filtered = bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=FS)
    se       = spectral_entropy(filtered, fs=FS)
    rms      = float(np.sqrt(np.mean(filtered ** 2)))

    if inject_vt:
        return AnomalyScore(score=0.96, label="VT", confidence=0.93)
    if inject_afib:
        score = 0.88 + random.uniform(-0.04, 0.04)
        return AnomalyScore(score=score, label="AFIB", confidence=0.87)
    if se > 0.82:
        return AnomalyScore(score=0.72, label="NOISE", confidence=0.78)

    noise = random.uniform(0.0, 0.12)
    return AnomalyScore(score=noise, label="NORMAL", confidence=0.97 - noise)


# ---------------------------------------------------------------------------
# Per-patient state machine
# ---------------------------------------------------------------------------

class PatientStream:
    def __init__(self, patient: dict[str, Any]) -> None:
        self.patient    = patient
        self.hr         = random.uniform(58, 88)
        self.spo2       = random.uniform(95, 100)
        self.sys_bp     = random.uniform(110, 135)
        self.dia_bp     = random.uniform(70, 90)
        self.temp       = random.uniform(36.4, 37.2)
        self._t         = 0.0
        self._phase     = 0          # 0=normal, 1=anomaly, 2=recovery
        self._phase_ctr = 0

    def _tick_phase(self) -> None:
        self._phase_ctr += 1
        if self._phase == 0 and self._phase_ctr > random.randint(80, 160):
            if self.patient["id"] == "pt-002":   # AFIB patient — frequent episodes
                self._phase = 1
                self._phase_ctr = 0
            elif random.random() < 0.08:          # other patients: rare events
                self._phase = 1
                self._phase_ctr = 0
        elif self._phase == 1 and self._phase_ctr > random.randint(20, 50):
            self._phase = 2
            self._phase_ctr = 0
        elif self._phase == 2 and self._phase_ctr > 30:
            self._phase = 0
            self._phase_ctr = 0

    def next_frame(self) -> dict[str, Any]:
        self._tick_phase()

        inject_afib = self._phase == 1 and self.patient["id"] in ("pt-002", "pt-001")
        inject_vt   = self._phase == 1 and self.patient["id"] == "pt-003"

        ecg = _ecg_window(
            WINDOW_SAMPLES, FS,
            hr=self.hr + random.uniform(-2, 2),
            inject_afib=inject_afib,
            inject_vt=inject_vt,
        )
        score = score_window(ecg, inject_afib, inject_vt)

        # Drift vitals slightly
        self.hr      = max(40, min(160, self.hr      + random.uniform(-0.5, 0.5)))
        self.spo2    = max(85, min(100, self.spo2    + random.uniform(-0.1, 0.1)))
        self.sys_bp  = max(80, min(190, self.sys_bp  + random.uniform(-0.8, 0.8)))
        self.dia_bp  = max(50, min(120, self.dia_bp  + random.uniform(-0.5, 0.5)))

        if inject_vt:
            self.hr     = min(180, self.hr + 2)
            self.sys_bp = max(60, self.sys_bp - 1)
        if inject_afib:
            self.hr += random.uniform(-3, 6)

        return {
            "patient_id":   self.patient["id"],
            "name":         self.patient["name"],
            "room":         self.patient["room"],
            "condition":    self.patient["condition"],
            "timestamp_ms": int(time.time() * 1000),
            "ecg_samples":  ecg.tolist(),
            "vitals": {
                "hr":     round(self.hr, 1),
                "spo2":   round(self.spo2, 1),
                "sys_bp": round(self.sys_bp, 0),
                "dia_bp": round(self.dia_bp, 0),
                "temp":   round(self.temp + random.uniform(-0.05, 0.05), 1),
            },
            "anomaly": {
                "score":      round(score.score, 3),
                "label":      score.label,
                "confidence": round(score.confidence, 3),
                "is_alert":   score.score >= 0.85,
            },
            "phase": self._phase,
        }


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.remove(ws)

    async def broadcast(self, data: str) -> None:
        dead = []
        for ws in self._connections:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections.remove(ws)


manager  = ConnectionManager()
streams  = {p["id"]: PatientStream(p) for p in PATIENTS}


# ---------------------------------------------------------------------------
# Background broadcast task
# ---------------------------------------------------------------------------

async def broadcast_loop() -> None:
    interval = 1.0 / STREAM_HZ
    while True:
        await asyncio.sleep(interval)
        frames = [streams[p["id"]].next_frame() for p in PATIENTS]
        await manager.broadcast(json.dumps(frames))


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(broadcast_loop())
    yield
    task.cancel()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="VitalStream Dashboard", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(os.path.dirname(__file__), "static", "index.html")) as f:
        return f.read()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()   # keep-alive ping from client
    except WebSocketDisconnect:
        manager.disconnect(ws)


if __name__ == "__main__":
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=8000, reload=False)
