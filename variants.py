"""Generate <=150 fundraising strategy variants."""
from itertools import product

ASK       = [250, 500, 1000]
FORMAT    = ["in-person", "virtual"]
TONE      = ["friendly", "formal", "urgent"]
CHANNEL   = ["email", "sms", "mail"]

variants = [
    dict(ask=a, format=f, tone=t, channel=c)
    for a, f, t, c in product(ASK, FORMAT, TONE, CHANNEL)
][:150]   # hard‑limit to 150 combos

if __name__ == "__main__":
    import json, pathlib
    path = pathlib.Path("variants.json")
    path.write_text(json.dumps(variants, indent=2))
    print(f"✅  Saved {len(variants)} variants →", path)