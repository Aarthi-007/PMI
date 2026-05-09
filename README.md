# AI-Powered Bayesian Postmortem Interval Estimation System

## Core Concept

The Postmortem Interval (PMI) Estimation System is an AI-assisted forensic intelligence engine that probabilistically estimates time since death by fusing multiple independent forensic evidence streams into a unified Bayesian posterior distribution.

Unlike traditional forensic workflows that rely on a single approximate estimate, the system models uncertainty explicitly and produces scientifically defensible probabilistic intervals.

The platform combines:

* thermodynamic body cooling models,
* AI-based postmortem image analysis,
* forensic pathology knowledge,
* and Bayesian statistical inference

to generate explainable and continuously updateable PMI predictions.

---

# System Architecture

The system operates using a multi-layer probabilistic fusion pipeline.

```
Evidence Acquisition
        ↓
Forensic Feature Extraction
        ↓
Independent Likelihood Modeling
        ↓
Bayesian Fusion Engine
        ↓
Posterior Distribution Generation
        ↓
Explainable Investigator Dashboard
```

---

# Layer 1 — Thermodynamic Body Cooling Model

The first evidence stream uses the Henssge Nomogram, a physics-based postmortem body cooling formulation widely used in forensic pathology.

The model estimates PMI by analyzing:

* core body temperature,
* ambient environmental temperature,
* body mass,
* clothing insulation,
* humidity,
* airflow,
* surface contact conditions,
* and environmental correction factors.

The system models postmortem cooling as exponential thermodynamic decay:

```
T(t) = T_a + (T_0 - T_a) × e^(-kt)
```

Where:

* T(t) = measured body temperature
* T_a = ambient temperature
* T_0 = initial physiological body temperature
* k = cooling coefficient dependent on environmental conditions

---

# Layer 2 — AI-Based Postmortem Change Classification

The second evidence stream analyzes postmortem physiological changes using deep learning.

The system processes forensic imagery to classify:

* livor mortis,
* rigor mortis,
* skin discoloration,
* and postmortem progression stages.

CNN-based computer vision models (ResNet-50, EfficientNet-B3) output softmax probability distributions across multiple PMI windows instead of rigid stage labels.

---

# Layer 3 — Bayesian Evidence Fusion Engine

The core intelligence layer combines all forensic evidence streams using Bayesian probabilistic fusion.

Core Bayesian inference concept:

```
P(TOD | Evidence) ∝ P(Evidence | TOD) × P(TOD)
```

This allows:

* conflicting evidence handling,
* uncertainty propagation,
* confidence interval generation,
* and probabilistic forensic reasoning.

---

# Installation

```bash
git clone https://github.com/Aarthi-007/PMI.git
cd PMI
pip install -r requirements.txt
```

---

# Quick Start

## Run CLI Demo

```bash
python pmi_fusion.py
```

Outputs 4 complete case examples with posterior PMI estimates and confidence intervals.

## Start FastAPI Server

```bash
uvicorn server:app --reload --port 8000
```

Access interactive API documentation at `http://localhost:8000/docs`

---

# API Endpoints

## POST /estimate

Compute PMI from forensic evidence.

**Request:**
```json
{
  "t_body_celsius": 31.5,
  "t_ambient_celsius": 21.0,
  "body_mass_kg": 75.0,
  "corrective_factor": 0.75,
  "livor_stage": "confluent",
  "rigor_stage": "partial"
}
```

**Response:**
```json
{
  "mode_hr": 11.2,
  "mean_hr": 11.8,
  "ci_68": [9.5, 13.8],
  "ci_95": [7.2, 16.4],
  "conflict": false,
  "conflict_spread_hr": 1.2,
  "stream_modes": {
    "henssge": 11.5,
    "image": 10.8
  }
}
```

## GET /factors

List Henssge environmental corrective factors.

## GET /stages

List valid postmortem change stages and PMI windows.

## GET /health

Service health check.

---

# Technical Stack

- **Core**: Python 3.8+
- **Bayesian**: SciPy, NumPy
- **API**: FastAPI, Uvicorn
- **ML**: PyTorch, TensorFlow (for CNN image analysis)

---

# Files

- `engine.py` — Core Bayesian PMI fusion engine with Henssge cooling model
- `pmi_fusion.py` — Alternative implementation with full Henssge 1988 formula and 4 demo cases
- `server.py` — FastAPI REST backend
- `requirements.txt` — Python dependencies
- `README.md` — This file

---

# Disclaimer

This system is intended for **forensic research and professional investigative use only**. All PMI estimates must be reviewed by qualified forensic pathologists and validated against independent evidence before use in legal proceedings. Individual biological variation, environmental factors, and postmortem redistribution effects create inherent uncertainty in all time-of-death estimations.

---

# Author

Created: 2026-05-09
License: MIT
