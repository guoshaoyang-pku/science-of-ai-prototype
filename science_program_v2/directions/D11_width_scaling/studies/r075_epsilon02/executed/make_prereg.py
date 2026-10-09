import hashlib,json
from pathlib import Path
root=Path('.').resolve(); study=root/'directions/D11_width_scaling/studies/r075_epsilon02'; base=root/'directions/D11_width_scaling/studies/r073_trace_one'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
files={}
for p in sorted((base/'results').iterdir()):
    if p.is_file(): files[str(p.relative_to(root))]={"sha256":sha(p),"mtime_ns":p.stat().st_mtime_ns,"size":p.stat().st_size}
files[str((base/'summary.json').relative_to(root))]={"sha256":sha(base/'summary.json'),"mtime_ns":(base/'summary.json').stat().st_mtime_ns,"size":(base/'summary.json').stat().st_size}
(study/'executed/source_manifest.json').write_text(json.dumps({"study":"r075_epsilon02","source_study":"r073_trace_one","files":files},ensure_ascii=False,indent=2)+'\n')
bfiles={}
for p in sorted((root/'directions/D11_width_scaling').rglob('*')):
    if p.is_file() and 'r075_epsilon02' not in str(p): bfiles[str(p.relative_to(root))]={"sha256":sha(p),"mtime_ns":p.stat().st_mtime_ns,"size":p.stat().st_size}
(study/'executed/baseline_manifest.json').write_text(json.dumps({"scope":"D11_width_scaling_prior_to_r075","files":bfiles},ensure_ascii=False,indent=2)+'\n')
cfg={"study":"r075_epsilon02","round":75,"direction":"D11_width_scaling","domain":"development","question":"复用r073 trace=1保存loss，将评价epsilon从0.01改为0.02，检验T/(d h_d)对log(50)/2与log(50)的误差边界。","threshold":0.02,"dimensions":[8,16,32,64,128,256,512,1024,2048,4096,8192],"coordinate_seeds":[0,1,2],"targets":["middle","endpoints"],"optimizer":{"eta":0.5,"momentum":0.0,"weight_decay":0.0,"initial_loss":0.5},"source_study":"directions/D11_width_scaling/studies/r073_trace_one","source_summary_sha256":sha(base/'summary.json'),"source_manifest_sha256":sha(study/'executed/source_manifest.json'),"pinned_sources":{"executed/run.py":sha(study/'executed/run.py'),"analysis.py":sha(study/'analysis.py')},"criteria":{"middle":0.03,"endpoints":0.05,"d_ge_128":0.003},"predictions":{"P1":"66/66保存loss读取成功，每个epsilon=.02首次步数可由loss数组复算，无缺失；只读评价不新增训练。","P2":"全网格middle的max |T/(d h_d)/(log(50)/2)-1| <= 0.03，endpoints <= 0.05；d>=128两目标均 <= 0.003。"},"known_before_registration":{"coefficients":{"middle":"log(50)/2=1.956011502714073","endpoints":"log(50)=3.912023005428146"},"provenance":"仅由epsilon=.02与r073已保存整数谱递推/损失曲线的注册前算术设定；本轮评价结果尚未读取。"},"boundaries":"仅固定特征、n=8192、非零样本d、lambda_i=i^-1/h_d、旧目标/坐标seed、float64全批量SGD的development；不推独立样本、learned width、lazy/rich、OOD或连续epsilon。","baseline_manifest":"directions/D11_width_scaling/studies/r075_epsilon02/executed/baseline_manifest.json"}
(study/'preregistration.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({"source_manifest_sha256":cfg["source_manifest_sha256"],"source_summary_sha256":cfg["source_summary_sha256"],"baseline_files":len(bfiles)},ensure_ascii=False))
