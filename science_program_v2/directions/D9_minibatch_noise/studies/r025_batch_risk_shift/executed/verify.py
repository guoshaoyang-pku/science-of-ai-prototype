import json, hashlib
from pathlib import Path
import numpy as np
STUDY=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 rec=json.loads((STUDY/'results/receipt.json').read_text()); out=[]
 for label,h in rec['cells'].items():
  p=STUDY/'results'/f'{label}.json'; q=p.with_suffix('.npz'); row=json.loads(p.read_text())
  with np.load(q) as z: finite=all(np.isfinite(z[k]).all() for k in z.files); lengths=len(z['mean_risk'])
  out.append({'label':label,'json_sha256':sha(p),'json_matches_receipt':sha(p)==h,'npz_sha256':sha(q),'finite':bool(finite),'length':lengths,'replicate_rows':int(z['replicate_risk'].shape[0])})
 result={'cells':len(out),'all_ok':all(x['json_matches_receipt'] and x['finite'] and x['length']==769 for x in out),'rows':out}
 (STUDY/'executed/verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'cells':result['cells'],'all_ok':result['all_ok']}))
if __name__=='__main__': main()
