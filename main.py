#!/usr/bin/env python3
"""AI-Donor unified launcher with interactive menu."""
import subprocess, sys, textwrap

task_map = {
    1: ["python", "gen_donor_dataset.py"],
    2: ["python", "build_rag_index.py",
        "--donor_csv", "output/donors_fake.csv",
        "--out_dir", "models"],
    3: ["python", "event_generate.py"],
    4: ["python", "search_events.py",
        "--index_dir", "models",
        "--events_json", "sample_events.json",
        "--top_k", "5"],
    7: ["python", "generate_kpis_report.py"],
    8: ["python", "grant_assistant.py"],
}

menu = textwrap.dedent("""
================ AI Donor CLI ================
1  Generate synthetic donor data
2  Build FAISS donor index
3  Generate synthetic events list (LLM)
4  Rank events by donor affinity
5  Match donors to events
6  Run KPI simulation (LLM heuristic)
7  Generate KPI report (PPTX)
8  Draft grant proposal
0  Exit
=============================================
Select option: """)

while True:
    try:
        choice = int(input(menu))
    except ValueError:
        print("Please enter a number."); continue
    if choice == 0:
        sys.exit(0)
    if choice == 5:
        event_id = input("Enter event ID to match (e.g., hk001): ").strip()
        cmd = [
            "python", "match_events.py",
            "--events_json", "sample_events.json",
            "--event_id", event_id,
            "--donor_csv", "output/donors_fake.csv",

            "--top_k", "5",
        ]
    elif choice == 6:
        event_id = input("Enter event ID for KPI simulation (e.g., hk001): ").strip()
        cmd = [
            "python", "simulate_kpis.py",
            "--events_json", "sample_events.json",
            "--event_id", event_id,
            "--donor_csv", "output/donors_fake.csv",
        ]
    else:
        cmd = task_map.get(choice)
        if not cmd:
            print("Invalid choice"); continue
    print("\n⇢ Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Error:", e)
    input("\nPress Enter to return to menu…")
