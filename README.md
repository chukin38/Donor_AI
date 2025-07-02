# Donor AI

This repository provides a set of small utilities for experimenting with fundraising data and event matching using LLM powered workflows.

## Features

- Generate synthetic donor profiles (`gen_donor_dataset.py`).
- Build a FAISS index of donors for similarity search (`build_rag_index.py`).
- Generate fundraising event data via an LLM API (`event_generate.py`).
- Search donors and events with sentence embeddings (`search_donors.py`, `search_events.py`).
- Simulate fundraising KPIs with the help of an LLM (`simulate_kpis.py`).
- Produce simple KPI reports (`generate_kpis_report.py`).
- Draft grant proposals (`grant_assistant.py`).
- A minimal FastAPI wrapper exposing some of the search functionality (`app.py`).

## Setup

Install Python 3.8+ and the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Below is a common workflow illustrating how the scripts tie together.

```bash
# 1. Generate donors
python gen_donor_dataset.py            # writes output/donors_fake.csv

# 2. Build the donor similarity index
python build_rag_index.py \
    --donor_csv output/donors_fake.csv \
    --out_dir models

# 3. (Optional) generate fundraising events via LLM
python event_generate.py --num_events 50 --out_file events_list.json

# 4. Rank events based on donor affinity
python search_events.py --index_dir models --events_json events_list.json

# 5. Search donors for a given text query
python search_donors.py \
    --index_dir models \
    --donor_csv output/donors_fake.csv \
    --query "community health" --top_k 5

# 6. Run KPI simulation for different strategy variants
python simulate_kpis.py

# 7. Generate a simple KPI PPTX report
python generate_kpis_report.py
```

To run the FastAPI service locally:

```bash
uvicorn app:API --reload
```

Generated artefacts such as FAISS indexes and reports are ignored via `.gitignore`.

