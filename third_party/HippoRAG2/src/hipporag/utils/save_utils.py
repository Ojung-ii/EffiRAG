# import os, csv, time
# import igraph as ig

# def _sanitize_id(x):
#     s = str(x)
#     return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

# def save_graph_with_qid(g: ig.Graph, query_id: str, filename: str, base_dir: str = "graph_debug"):
#     qid = _sanitize_id(query_id)
#     out_dir = os.path.join(base_dir, f"q_{qid}")
#     os.makedirs(out_dir, exist_ok=True)
#     path = os.path.join(out_dir, filename)
#     g.write_graphml(path)
#     print(f"[SAVE] {path}")
#     return path

# def append_manifest_row(query_id: str, row: dict, base_dir: str = "graph_debug"):
#     qid = _sanitize_id(query_id)
#     out_dir = os.path.join(base_dir, f"q_{qid}")
#     os.makedirs(out_dir, exist_ok=True)
#     path = os.path.join(out_dir, "manifest.csv")
#     write_header = not os.path.exists(path)
#     with open(path, "a", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=["file","type","root_id","nodes","edges","timestamp"])
#         if write_header: w.writeheader()
#         w.writerow({**row, "timestamp": int(time.time())})

import os, csv, igraph as ig
from pathlib import Path

def _sanitize_id(x):
    s = str(x or "")
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

def save_graph_with_ids(
    g: ig.Graph,
    query_id: str,
    run_id: str,
    filename: str,
    base_dir: str = "graph_debug",
):
    qid, rid = _sanitize_id(query_id), _sanitize_id(run_id)
    out_dir = Path(base_dir) / f"q_{qid}" / rid
    out_dir.mkdir(parents=True, exist_ok=True)

    # 그래프 자체에도 메타 심기(나중에 파일만 봐도 매핑 가능)
    g["query_id"] = qid
    g["run_id"]   = rid

    path = out_dir / filename
    g.write_graphml(str(path))
    print(f"[SAVE] {path}")
    return str(path)

def append_manifest_row(
    query_id: str,
    run_id: str,
    row: dict,
    base_dir: str = "graph_debug",
):
    qid, rid = _sanitize_id(query_id), _sanitize_id(run_id)
    out_dir  = Path(base_dir) / f"q_{qid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.csv"

    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        fields = ["query_id","run_id","file","type","root_id","nodes","edges","timestamp"]
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header: w.writeheader()
        w.writerow({
            "query_id": qid,
            "run_id": rid,
            **row,
            "timestamp": int(time.time()),
        })

import time, hashlib

def make_qid_from_text(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode()).hexdigest()[:10]

def now_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns()%1_000_000:06d}"
