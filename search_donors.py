#!/usr/bin/env python3
import argparse, numpy as np, faiss, pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Search donors by text query')
    ap.add_argument('--index_dir', required=True, help='Folder containing donor_vectors.faiss and donor_ids.npy')
    ap.add_argument('--donor_csv', required=False, default='donors_1k.csv', help='CSV for lookup')
    ap.add_argument('--query', required=True, help='Text query for event or criteria')
    ap.add_argument('--top_k', type=int, default=5, help='Number of donors to retrieve')
    args = ap.parse_args()

    # Load index and ids
    index = faiss.read_index(f"{args.index_dir}/donor_vectors.faiss")
    donor_ids = np.load(f"{args.index_dir}/donor_ids.npy")

    # Encode query
    model = SentenceTransformer(MODEL_NAME)
    q_vec = model.encode([args.query])
    faiss.normalize_L2(q_vec)

    # Search
    D, I = index.search(q_vec, args.top_k)
    df = pd.read_csv(args.donor_csv) if args.donor_csv else None

    # Output
    print(f"Top {args.top_k} donors for query '{args.query}':")
    for score, idx in zip(D[0], I[0]):
        donor_info = df.iloc[donor_ids[idx]].to_dict() if df is not None else f"ID {donor_ids[idx]}"
        print(f"- Score: {score:.4f}, Donor: {donor_info}")