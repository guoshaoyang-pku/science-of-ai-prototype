from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
STUDY = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pinned_analysis', STUDY / 'analysis.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
try:
    analysis.main()
except TypeError as error:
    summary = json.loads((STUDY / 'summary.json').read_text())
    if str(error) != "'NoneType' object is not subscriptable" or summary['primary_mixed_relative_error'] is not None:
        raise
    print('Partial-cell scientific audit saved; no primary-scale output statistic yet.')
summary = json.loads((STUDY / 'summary.json').read_text())
for name in ('primary_mixed_relative_error', 'all_scale_mixed_relative_error', 'primary_no_cross_relative_error'):
    if summary[name] is not None:
        summary[name].pop('seed_95pct_t_interval', None)
        summary[name]['sampling_note'] = '固定bias×scale×seed网格描述统计；无pooled iid/seed置信区间'
summary['reporting_correction'] = {
    'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'pinned_scientific_analysis_unchanged': True,
    'reason': '原pinned stats在跨格点误差描述中套用seed CI；删去三个无采样解释的pooled区间。每delta六seed配对CI及六seed不对称差CI保留。首cell空统计打印错误有条件处理。R、预测、阈值和每seed数据均不改。'
}
(STUDY / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
