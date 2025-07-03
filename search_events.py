#!/usr/bin/env python3
import argparse, json, numpy as np, faiss, pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def event_to_text(e):
    return (
        f"{e.get('Event_Name', e.get('title',''))} | "
        f"Cause: {e.get('Cause_Focus', e.get('cause',''))} | "
        f"Goal: HK${e.get('Goal_Amount', e.get('goal_amount','N/A'))} | "
        f"Date: {e.get('Event_Date', e.get('date_start',''))}"
    )

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Rank events by aggregate donor similarity')
    ap.add_argument('--index_dir', required=True, help='Folder with donor_vectors.faiss & donor_ids.npy')
    ap.add_argument('--events_file', required=True, help='CSV or JSON file with events list')
    ap.add_argument('--top_k', type=int, default=5, help='Number of top events to return')
    ap.add_argument('--metric', choices=['mean','sum','count'], default='mean',
                   help="Aggregation metric: mean cosine, sum cosine, or count>0.5 similarity")
    args = ap.parse_args()

    # Load donors index
    index = faiss.read_index(f"{args.index_dir}/donor_vectors.faiss")
    donor_ids = np.load(f"{args.index_dir}/donor_ids.npy")
    n_donors = len(donor_ids)

    # Load events
    if args.events_file.endswith('.csv'):
        events = pd.read_csv(args.events_file).to_dict(orient='records')
    else:
        events = json.load(open(args.events_file, 'r', encoding='utf-8'))

    # Embed events
    model = SentenceTransformer(MODEL_NAME)
    texts = [event_to_text(e) for e in events]
    vecs = model.encode(texts, batch_size=32, show_progress_bar=False)
    faiss.normalize_L2(vecs)

    # Score each event
    results = []
    for e, vec in zip(events, vecs):
        D, I = index.search(vec.reshape(1, -1), n_donors)
        sims = D[0]
        if args.metric == 'mean':
            score = float(sims.mean())
        elif args.metric == 'sum':
            score = float(sims.sum())
        else:  # count of sims > 0.5
            score = int((sims > 0.5).sum())
        results.append({**e, 'score': round(score,4)})

    # Sort and output
    results = sorted(results, key=lambda x: x['score'], reverse=True)[:args.top_k]
    print(f"Top {args.top_k} events by donor similarity ({args.metric}):")
    for ev in results:
        name = ev.get('Event_Name', ev.get('title',''))
        print(f"- {name} (Score: {ev['score']})")
