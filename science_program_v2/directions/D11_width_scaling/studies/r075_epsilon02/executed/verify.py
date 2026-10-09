import hashlib,json,math
from pathlib import Path
import numpy as np
STUDY=Path(__file__).resolve().parent.parent; ROOT=STUDY.parents[3]; BASE=ROOT/'directions/D11_width_scaling/studies/r073_trace_one'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 cfg=json.loads((STUDY/'preregistration.json').read_text()); rec=json.loads((STUDY/'results/receipt.json').read_text()); rows=[]
 for name,digest in rec['cells'].items():
  out=STUDY/'results'/(name+'.json'); assert sha(out)==digest
  row=json.loads(out.read_text()); srcj=BASE/'results'/(name+'.json'); srcn=BASE/'results'/(name+'.npz')
  assert row['source_json_sha256']==sha(srcj) and row['source_npz_sha256']==sha(srcn)
  with np.load(srcn) as z:
   loss=np.asarray(z['loss'],dtype=float); hit=int(np.flatnonzero(loss/loss[0]<=.02)[0]); assert hit==row['measured_step_epsilon002']
  rows.append(row)
 assert len(rows)==66
 imm=[]
 for rel,entry in json.loads((STUDY/'executed/baseline_manifest.json').read_text())['files'].items():
  p=ROOT/rel; imm.append(sha(p)==entry['sha256'] and p.stat().st_mtime_ns==entry['mtime_ns'])
 assert all(imm)
 print(json.dumps({'all_passed':True,'saved_cells':len(rows),'all_source_hashes_match':True,'all_threshold_hits_match':True,'prior_files_unchanged':len(imm),'prior_files_all_unchanged':all(imm)}))
 (STUDY/'executed/independent_verification.json').write_text(json.dumps({'all_passed':True,'saved_cells':len(rows),'all_source_hashes_match':True,'all_threshold_hits_match':True,'prior_files_unchanged':len(imm),'prior_files_all_unchanged':all(imm)},indent=2)+'\n')
if __name__=='__main__': main()
