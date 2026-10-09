"""Merge current direction/study evidence and verify process identities."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from experiment import save

ROOT = Path(__file__).resolve().parent


def identity(pid):
    if not pid: return None
    p = subprocess.run(["ps","-p",str(pid),"-o","lstart=","-o","stat=","-o","command="],capture_output=True,text=True,timeout=5)
    fields=p.stdout.strip().split(None,6)
    if p.returncode or len(fields)<7 or "Z" in fields[5]: return None
    return {"pid":pid,"started":" ".join(fields[:5]),"command":fields[6]}


def refresh():
    directions={}
    for path in sorted((ROOT/"directions").glob("*/state.json")):
        directions[path.parent.name]=json.loads(path.read_text())
    studies={}
    for path in sorted((ROOT/"studies").glob("*/current.json")):
        s=json.loads(path.read_text());pid=s.get("pid") or (s.get("process") or {}).get("pid")
        live=identity(pid)
        s["process_live_now"]=live
        s["checked_at"]=time.time()
        studies[path.parent.name]=s
    release_path=ROOT/"state/release_status.json"
    release=json.loads(release_path.read_text()) if release_path.exists() else {}
    value={"status":release.get("status","directed_science_active"),"first_chain_complete":(ROOT/"studies/A01_head_mediation/analysis.json").exists(),
           "checked_at":time.time(),"directions":directions,"studies":studies,
           "kb":json.loads((ROOT/"kb.json").read_text()),"solver_evaluation":"not_run",
           "release":release,
           "delivery":{"mode":"state file; waiting for new objective after release" if release else "state file","current_turn_steer_available":False,
                       "automatic_probe_received":False,"new_queue_submissions":0},
           "old_publisher":identity(15910),"refresh_process":identity(os.getpid())}
    save(ROOT/"state/current.json",value)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--watch",action="store_true");parser.add_argument("--interval",type=float,default=30);args=parser.parse_args()
    while True:
        s=refresh()
        if not args.watch:
            print(json.dumps({"directions":len(s["directions"]),"studies":len(s["studies"]),
                              "live":[name for name,x in s["studies"].items() if x["process_live_now"]]}))
            return
        time.sleep(args.interval)


if __name__=="__main__":main()
