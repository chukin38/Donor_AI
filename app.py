from fastapi import FastAPI
import subprocess, json

API = FastAPI(title="AI Donor API")

def run(cmd:list):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode!=0: return {"error": r.stderr}
    return {"output": r.stdout}

@API.get("/search/events")
def rank_events(top_k:int=5):
    return run(["python", "search_events.py", "--index_dir","models","--events_json","events_list.json","--top_k", str(top_k)])

@API.get("/search/donors")
def donors(q:str, top_k:int=5):
    return run(["python","search_donors.py","--index_dir","models","--donor_csv","donors_1k.csv","--query",q,"--top_k",str(top_k)])

@API.get("/match")
def match(top_k:int=5, metric:str="mean"):
    return run(["python","match_events.py","--events_json","events_list.json","--donor_csv","donors_1k.csv","--index_dir","models","--top_k",str(top_k),"--metric",metric])