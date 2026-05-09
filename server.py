"""
PMI System — FastAPI Backend
============================
REST API wrapping the Bayesian fusion engine.

Run:
    pip install fastapi uvicorn numpy scipy
    uvicorn server:app --reload --port 8000

Endpoints:
    POST /estimate       — compute PMI from JSON payload
    GET  /factors        — list Henssge corrective factors
    GET  /stages         — list valid livor/rigor stages
    GET  /health         — service health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from typing import Optional
import uvicorn

from engine import (
    estimate_pmi,
    CORRECTIVE_FACTORS,
    LIVOR_WINDOWS,
    RIGOR_WINDOWS,
    PMI_GRID,
)

app = FastAPI(
    title="PMI Bayesian Fusion System",
    description="Postmortem interval estimation via fused forensic evidence streams",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class EstimateRequest(BaseModel):
    # Henssge inputs
    t_body_celsius:    float = Field(..., ge=0,  le=42,  description="Rectal temperature at scene (°C)")
    t_ambient_celsius: float = Field(..., ge=-20, le=50, description="Ambient scene temperature (°C)")
    body_mass_kg:      float = Field(..., ge=10, le=300, description="Body mass (kg)")
    corrective_factor: float = Field(1.0, ge=0.1, le=3.0, description="Henssge environmental Cf")
    henssge_sigma:     float = Field(2.0, ge=0.5, le=8.0, description="Cooling model base uncertainty (hours)")

    # Hard stage labels
    livor_stage: Optional[str] = Field(None, description="none | faint | confluent | fixed")
    rigor_stage: Optional[str] = Field(None, description="none | partial | full | resolving")

    # CNN softmax vectors (override hard labels when present)
    livor_softmax: Optional[dict[str, float]] = Field(None, description="CNN softmax over livor stages")
    rigor_softmax: Optional[dict[str, float]] = Field(None, description="CNN softmax over rigor stages")

    use_henssge: bool = True
    use_image:   bool = True

    @model_validator(mode="after")
    def check_evidence(self):
        if not self.use_henssge and not self.use_image:
            raise ValueError("At least one evidence stream must be enabled.")
        if self.livor_stage and self.livor_stage not in LIVOR_WINDOWS:
            raise ValueError(f"Invalid livor_stage. Valid: {list(LIVOR_WINDOWS)}")
        if self.rigor_stage and self.rigor_stage not in RIGOR_WINDOWS:
            raise ValueError(f"Invalid rigor_stage. Valid: {list(RIGOR_WINDOWS)}")
        return self


class EstimateResponse(BaseModel):
    mode_hr:            float
    mean_hr:            float
    ci_68:              list[float]
    ci_95:              list[float]
    conflict:           bool
    conflict_spread_hr: float
    stream_modes:       dict[str, float]
    henssge_mu:         float
    pmi_grid:           list[float]
    posterior:          list[float]
    henssge_lik:        list[float]
    image_lik:          list[float]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "grid_points": len(PMI_GRID), "pmi_max_hr": 72.0}


@app.get("/factors")
def get_factors():
    """List all named Henssge corrective factors."""
    return {
        "corrective_factors": CORRECTIVE_FACTORS,
        "description": (
            "Values > 1 = faster cooling (wet, windy, immersed). "
            "Values < 1 = slower cooling (clothed, wrapped). "
            "1.0 = naked in still air (reference)."
        ),
    }


@app.get("/stages")
def get_stages():
    """List valid postmortem change stages with PMI window hints."""
    return {
        "livor_stages": {
            k: {"peak_hr": v[0], "sigma_lo": v[1], "sigma_hi": v[2]}
            for k, v in LIVOR_WINDOWS.items()
        },
        "rigor_stages": {
            k: {"peak_hr": v[0], "sigma_lo": v[1], "sigma_hi": v[2]}
            for k, v in RIGOR_WINDOWS.items()
        },
    }


@app.post("/estimate", response_model=EstimateResponse)
def estimate(req: EstimateRequest):
    """
    Compute PMI posterior from fused evidence streams.

    Accepts either hard stage labels (livor_stage / rigor_stage)
    or CNN softmax dicts (livor_softmax / rigor_softmax).
    Softmax takes precedence when both are provided.
    """
    try:
        result = estimate_pmi(
            t_body_celsius    = req.t_body_celsius,
            t_ambient_celsius = req.t_ambient_celsius,
            body_mass_kg      = req.body_mass_kg,
            corrective_factor = req.corrective_factor,
            henssge_sigma     = req.henssge_sigma,
            livor_stage       = req.livor_stage,
            rigor_stage       = req.rigor_stage,
            livor_softmax     = req.livor_softmax,
            rigor_softmax     = req.rigor_softmax,
            use_henssge       = req.use_henssge,
            use_image         = req.use_image,
        )
        return result.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine error: {e}")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
