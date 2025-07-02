#!/usr/bin/env python3
"""Simple heuristic donor matching for a selected event."""

import argparse
import json
import pandas as pd


def load_event(events, event_id: str) -> dict:
    for ev in events:
        if ev.get("event_id") == event_id:
            return ev
    raise ValueError(f"Event id {event_id} not found")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Match donors to an event by cause")
    ap.add_argument("--events_json", required=True, help="Path to events JSON")
    ap.add_argument("--event_id", required=True, help="ID of the event")
    ap.add_argument("--donor_csv", required=True, help="CSV of donors")
    ap.add_argument("--top_k", type=int, default=5, help="Number of donors to show")
    args = ap.parse_args()

    events = json.load(open(args.events_json, "r", encoding="utf-8"))
    event = load_event(events, args.event_id)

    donors = pd.read_csv(args.donor_csv)
    matches = donors[donors["primary_cause"].str.lower() == str(event.get("cause", "")).lower()]
    if len(matches) < args.top_k:
        remaining = donors.drop(matches.index)
        extra = remaining.sample(args.top_k - len(matches)) if len(remaining) >= args.top_k - len(matches) else remaining
        matches = pd.concat([matches, extra])
    matches = matches.head(args.top_k)

    print(f"Top {args.top_k} donors for event '{event['title']}' ({event['event_id']}):")
    for _, row in matches.iterrows():
        print(f"- {row['name']} | Cause: {row['primary_cause']} | Email: {row['communication_pref']}")
