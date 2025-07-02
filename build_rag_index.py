#!/usr/bin/env python3
"""
build_index.py
--------------
用法：
    python build_index.py --donor_csv donors_1k.csv \
                          --out_dir  models \
                          --model   sentence-transformers/all-MiniLM-L6-v2
"""
import argparse, json, os, numpy as np, pandas as pd, faiss
from sentence_transformers import SentenceTransformer


def donor_to_text(row: pd.Series) -> str:
    """把一列 Donor 轉成語意敘述文字"""
    return (
        f"{row.full_name}, age {row.age}, {row.religion} donor from {row.state}. "
        f"Primary cause: {row.primary_cause}. Lifetime donated ${row.lifetime_donation_usd} "
        f"with average gift ${row.average_gift_usd}. Major gift score {row.major_gift_score}/100."
    )


def build_index(csv_path: str, out_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"📥  Loading donors from {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"📝  Converting {len(df)} donors to text …")
    texts = df.apply(donor_to_text, axis=1).to_list()

    print(f"🔄  Encoding with model `{model_name}`")
    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, batch_size=64, show_progress_bar=True)

    # 建立 FAISS Index
    dim = vecs.shape[1]
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(dim)  # Cosine（內積）
    index.add(vecs)

    index_path = os.path.join(out_dir, "donor_vectors.faiss")
    id_path = os.path.join(out_dir, "donor_ids.npy")
    faiss.write_index(index, index_path)
    np.save(id_path, df.index.values)

    print(f"✅  Saved index  →  {index_path}")
    print(f"✅  Saved id map →  {id_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build FAISS index for donor vectors")
    ap.add_argument("--donor_csv", required=True, help="Path to donors CSV")
    ap.add_argument("--out_dir", default="models", help="Directory to save index files")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model name")
    args = ap.parse_args()
    build_index(args.donor_csv, args.out_dir, args.model)
