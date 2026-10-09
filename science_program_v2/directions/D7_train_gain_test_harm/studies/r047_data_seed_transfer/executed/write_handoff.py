from pathlib import Path
import json

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[3]
DIRECTION = STUDY.parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def ci(value):
    lo, hi = value['seed_95pct_t_interval']
    return f"{value['mean']:.6f} [{lo:.6f}, {hi:.6f}]"


def main():
    summary = json.loads((STUDY/'summary.json').read_text())
    independent = json.loads((STUDY/'executed/independent_verification.json').read_text())
    commit = summary['preregistration_commit']
    primary_table = '| 函数×宽度 | mean(H0) | mean(H1) | mean(D) [95%初始化区间] | D>0 | 新−旧data的mean(D) [配对初始化区间] |\n|---|---:|---:|---:|---:|---:|\n'
    for unit in summary['units']:
        m = unit['metrics']
        primary_table += f"| {unit['function']}×{unit['width']} | {m['clean_test_gap']['mean']:.6f} | {m['noisy_test_gap']['mean']:.6f} | {ci(m['noise_amplification'])} | {unit['positive_amplification_seeds']}/3 | {ci(m['new_minus_old_D'])} |\n"
    pairs = '| 函数×宽度 | init | 新D | 旧data同noise的D | 新−旧D |\n|---|---:|---:|---:|---:|\n'
    for p in summary['paired_seed_results']:
        pairs += f"| {p['function']}×{p['width']} | {p['seed']} | {p['noise_amplification']:.6f} | {p['previous_data_amplification']:.6f} | {p['new_minus_old_D']:.6f} |\n"
    late = '| 函数×宽度 | σ | 真实C均值 [95%初始化区间] | 真实C>0 | 切线C均值 [95%初始化区间] | 切线C>0 |\n|---|---:|---:|---:|---:|---:|\n'
    for unit in summary['late_units']:
        real, tangent = unit['metrics']['real'], unit['metrics']['tangent']
        late += f"| {unit['function']}×{unit['width']} | {unit['sigma']} | {ci(real['late_test_change'])} | {real['positive_late_test_seeds']}/3 | {ci(tangent['late_test_change'])} | {tangent['positive_late_test_seeds']}/3 |\n"
    finding = f'''# 新data7339下，相对test差距放大判据保持

程序第47轮、D7第4轮只检验一个development问题：固定noise733123及原recipe，只将data seed7331改为预指定7339，原.02、2/3、3/4判据是否保持。24/24新cell、48轨迹完成，累计训练{summary['training_seconds']:.6f}秒，P1 supported 4/4，12/12 D>0。没有复用旧clean轨迹作为新data对照，也没有重跑或覆盖旧36cell。

## 合同与提交时序

d5 Gaussian train32/test256，两函数mixed_sine/radial、含bias两hidden SiLU、width8/32、init11/29/47；默认float32 Linear初始化后double，CPU float64，full-batch halfMSE、SGD eta=.05/mom0/nodecay/T512。clean目标使用train均值与ddof0标准差，train_y=clean_y+sigma epsilon，sigma0/1；PCG64 noise733123的同一epsilon跨函数/宽度/init共用，不center、不rescale、不在加噪后归一化，test始终clean。检查点0/1/8/32/128/256/512。

先核验旧r023科学收尾09388db及final_commit_verification，r015真正收尾8fa680d、0c64176仅收据；120历史文件hash/mtime先冻结，36历史成功cell只读。旧报告训练前改为固定六节，旧数字与科学结论全部保留；不存在inbox.md。5aa2a41同提交锁定新seed、数值预测、run.py、analysis.py及旧引用hash，先于全部训练，runner逐字核验后记录gate。首cell分析通过后才运行其余23cell；首cellJSON/NPZ hash及mtime不变。

H_sigma为第512步真实减切线的干净test halfMSE，D=H1−H0在同init配对。唯一P1要求至少3/4函数×宽度单元mean(D)>=.02且各至少2/3 init D>0；24新cell齐全且核验后才判定。三init不是三种独立recipe。95% Student-t区间df2仅描述固定data/noise下初始化差异，不含跨data/noise不确定性。

## 预测与逐初始化结果

{primary_table}

radial×8的D区间跨零，原判据不要求区间排除零。固定noise下跨data的四个配对差区间全部跨零；前三个非radial×8均值上升，但mixed_sine×8 init29下降.334783、mixed_sine×32 init11下降.017555。radial×8均值下降.152092，init29反而增加.019018。不宣称跨data单调性或稳定幅度。

{pairs}

## 终点与晚期分别记录

C=同模型第512步减第128步的test halfMSE，正值表示晚期test上升。这是描述诊断，没有新成功判据。全部12新noisy H1>0，但radial×32 init47 noisy仍H1=.036880343842065455、真实C=−.021062709533282042；该单元真实晚期上升只有2/3 init。新noisy12/12 train增益G>0，单元均值.433225–.573030；更低带噪train loss不等于更接近clean目标。

{late}

旧反例保留：mixed_sine×32 init11 clean H=.020497而真实C=−.001370；noise733107的radial×32 init29 noisy H=.012378而真实C=−.029205；旧data7331/noise733123 radial×32的3/3 D>0却3/3 H1<0且真实C<0。旧noise733123的radial×8 init11比noise733107增加.353716以及三个跨零区间均保留。

## 保存证据核验与边界

60cell（旧36+新24）数据、finite、request/NPZ与对应提交字节通过，初始参数、完整train/test Jacobian、初始和终点真实输出重建误差0。einsum切线全部train loss及所有train/test检查点最大误差：新24cell单独6.161738e−15，新旧60cell合计7.147061e−15；旧独立复算<7e−15的历史结论保留，不合并篡改。旧matmul警告全文hash不变、根因未定；本轮training/analysis stderr为空。

独立verify_saved.py直接NPZ复算D/H/跨data差，均值与逐seed数值完全一致；df2区间用解析分位数，与SciPy区间最大差1.097433e−11。首次核验以1e−12比较区间而失败，原源码与错误保留；仅独立区间核验容差改为1e−10，原科学阈值、pinned分析和结果均未改，均值/D核验仍1e−12。无新训练的恢复24/24成功cell核验通过，120旧文件与48新成功文件hash/mtime全部不变。

换data同时改变train/test输入、clean目标与train归一化；保持参数/noise相同仅控制其余因子，不能把总效应归为一个机制。结果支持一个额外预指定data draw上的相对噪声敏感度差，不能估计跨data/noise概率或机制比例，不形成train-only早停规则，不外推其他sigma、optimizer、LN、深度或数据规模。

下一小问题建议：保持data7339/noise733123及原函数/宽度/init/SGD/T512，仅新增sigma=.5的12cell，复用本轮sigma0/1轨迹，检验D(.5)=H(.5)−H(0)是否至少3/4单元mean>=.01且各>=2/3 init>0。先保存明确预测、新pinned源码与本轮引用hash并commit，匹配后才训练；这不预测线性/二次噪声缩放或连续阈值。

## 证据

- [预注册](../studies/r047_data_seed_transfer/preregistration.json)
- [配对汇总](../studies/r047_data_seed_transfer/summary.json)
- [60cell保存证据核验](../studies/r047_data_seed_transfer/executed/saved_evidence_verification.json)
- [独立复算与不覆盖恢复](../studies/r047_data_seed_transfer/executed/independent_verification.json)
- [旧科学收尾审核](../studies/r047_data_seed_transfer/executed/prior_closeout_audit.json)
- [独立区间首失败](../studies/r047_data_seed_transfer/executed/independent_verification_first_failure.json)
'''
    (DIRECTION/'findings/r047_data_seed_transfer.md').write_text(finding)
    report_path=DIRECTION/'report.md'
    report=report_path.read_text()
    report=report.replace('实测边界：d5 Gaussian', '实测边界：d5 Gaussian',1)
    boundary='在相同recipe、noise733123下，仅改data为7339仍4/4通过，mean(D)=.111892–.519513、12/12 D>0；radial×8的95%初始化区间跨零。'
    report=report.replace('新draw三个95%init区间跨零。','新draw三个95%init区间跨零。'+boundary,1)
    report=report.replace('s为初始化seed；ν为噪声seed；t为优化步数。','s为初始化seed；ν为噪声seed；t为优化步数；下式数据条件逐data seed计算，data seed决定train/test输入与clean目标归一化。',1)
    new_phenomenon=f'''固定noise733123，更换data7339后，四个函数×宽度单元仍满足原.02、2/3、3/4判据。新旧data的四个D配对差区间全部跨零，结果只支持本次条件验证。更换data共同改变输入、clean目标与train归一化，不能拆出单一机制。

{primary_table}

所有12个新noisy终点H1为正，真实网络对带噪train标签的G单元均值.433225–.573030、12/12 G>0；终点相对劣势仍不保证同模型晚期test上升。以下新data晚期诊断与旧data分开：

{late}

此前两个噪声draw均使用data7331，已测结果如下。

'''
    report=report.replace('## 现象与解释\n\n','## 现象与解释\n\n'+new_phenomenon,1)
    report=report.replace('## 失败与反例\n\n','''## 失败与反例

新data的radial×8 D区间[−.081196,.304979]跨零。mixed_sine×8/32均值D增大却分别有init29/11下降.334783/.017555；radial×8均值下降却init29增加.019018。全部跨data配对差区间跨零，均值方向不作单调规律。

新data的radial×32 init47 noisy：H1=.036880344>0，真实C=−.021062710<0。终点劣势与晚期下降仍可共存；同单元真实C>0仅2/3 init。

独立核验首次用1e−12比较解析df2与SciPy数值区间失败，差异最大1.097433e−11；保留原错误/源码，仅独立区间比较容差放至1e−10，原分析、训练合同与科学判据保持。

''',1)
    report=report.replace('固定一个data、两个noise draw，不外推新数据/噪声','上述旧噪声对照固定一个data、两个noise draw，不外推新数据/噪声',1)
    method=f'''\n新data使用相同d5 Gaussian train32/test256、noise733123、函数/宽度/init及SGD，共24新cell、48轨迹、4recipe、12初始化配对；累计训练{summary['training_seconds']:.6f}秒。sigma0/1在新data精确共享输入/clean目标/归一化/参数/J，跨data精确共享初始参数与noise。所有区间为df2初始化区间，跨data仍按同init作差，不把两个data draw作概率样本。

旧36与新24合计60cell核验通过；新24cell独立切线误差最大6.161738e−15、全60cell最大7.147061e−15，参数/J及真实输出重建误差0。120历史与48新成功文件hash/mtime不变，恢复0新训练、24/24只核验。训练前同提交合同及pinned源码匹配，不覆盖旧成功cell。\n'''
    report=report.replace('## 未决问题\n',method+'\n## 未决问题\n',1)
    before,tail=report.split('## 未决问题\n',1); _,evidence=tail.split('## 证据\n',1)
    next_text='仅新增sigma=.5，保持data7339/noise733123与原函数/宽度/init/SGD/T512，复用已保存sigma0/1，检验至少3/4 mean(D(.5))>=.01且各>=2/3 init D(.5)>0。先冻结数值预测、源码与引用hash并提交，再运行12新cell；不作线性/二次缩放或连续阈值推断。机制比例、跨data/noise概率和train-only早停仍未建立。D7负责损害机制边界，D2负责U型曲线与早停。'
    report=before+'## 未决问题\n\n'+next_text+'\n\n## 证据\n'+evidence
    report+='\n- [新data预测与结果](findings/r047_data_seed_transfer.md)\n- [新data预注册](studies/r047_data_seed_transfer/preregistration.json)\n- [新data逐初始化汇总](studies/r047_data_seed_transfer/summary.json)\n- [新data独立复算与不覆盖恢复](studies/r047_data_seed_transfer/executed/independent_verification.json)\n- [新data保存证据核验](studies/r047_data_seed_transfer/executed/saved_evidence_verification.json)\n- [预注册提交5aa2a41](../../.git)；旧科学收尾09388db、8fa680d及收据0c64176见[历史收尾审核](studies/r047_data_seed_transfer/executed/prior_closeout_audit.json)。\n'
    report_path.write_text(report)
    kb_path=REPO/'central/kb.json'; kb=json.loads(kb_path.read_text())
    scope='d5 Gaussian train32/test256、data7339/noise733123、mixed_sine/radial×width8/32、两hidden SiLU含bias、init11/29/47、CPU float64全批量SGD eta=.05/mom0/nodecay/T512、train归一化后sigma0/1'
    evidence=[str((STUDY/'summary.json').relative_to(REPO)),str((STUDY/'executed/independent_verification.json').relative_to(REPO))]
    claims=[{'id':'D7-005','direction':'D7','round':47,'text':scope+'下，噪声扩大真实相对切线终点test差距：四meanD=.469371/.519513/.111892/.186296，12/12初始化D>0，原判据4/4通过；radial×8的95%初始化区间[−.081196,.304979]跨零。','evidence':evidence,'domain':'development','status':'measured','boundary':'单预指定新data draw；三init区间不含跨data/noise不确定性；更换data共同改变输入/clean目标/归一化，不拆机制比例或提供train-only早停。'},
            {'id':'D7-006','direction':'D7','round':47,'text':scope+'的radial×32 init47 sigma1，终点真实减切线test H=.036880344>0，但真实网络128→512的test变化C=−.021062710<0；该单元真实晚期上升仅2/3 init。','evidence':evidence,'domain':'development','status':'measured','boundary':'单cell反例只区分终点比较与同模型晚期变化；无新预测，不能作早停阈值或未测条件因果外推。'}]
    assert not any(c['id'] in {x['id'] for x in kb['claims']} for c in claims)
    kb['claims'].extend(claims); dump(kb_path,kb)
    state_path=REPO/'central/state.json'; state=json.loads(state_path.read_text()); before_state=json.loads((STUDY/'executed/state_before_round.json').read_text())
    direction='D7_train_gain_test_harm'
    state['directions'][direction]['last_round']={'round':47,'direction':direction,'result':'ok','summary':'24/24新cell；data7339固定noise733123的P1 supported4/4、12/12 D>0，radial×8区间跨零；跨data四差区间跨零，新H>0/C<0反例保留，证据与收尾待核验。'}
    state['directions'][direction]['next_question']='r047完成24/24新sigma0/1cell，48轨迹，预注册5aa2a41同提交锁定seed/源码先于训练；先核验科学收尾与final_commit_verification，不重跑/覆盖r00724、r02312与新24成功cell。data7339固定noise733123的P1 supported4/4，12/12 D>0，四meanD=.469371/.519513/.111892/.186296，radial×8 CI[−.081196,.304979]跨零；四跨data同init差CI全部跨零，不作幅度单调。新radial×32 init47 noisy H=.036880344且真实C=−.021062710，晚期上升仅2/3；旧两分歧、旧radial32三H1/C负和radial8init11+.353716保留。旧120与新48文件hash/mtime不变，60cell参数/J/真实输出重建0、独立新切线6.161738e−15/全60cell7.147061e−15，与旧<7e−15分开；旧matmul根因未定。独立df2解析区间首次1e−12比较失败，保存原错误/源码后仅核验容差1e−10，科学判据/pinned源码不改。下一小问题建议同data7339/noise733123/d5/train32/test256/函数宽度init/SGDeta.05mom0nodecay/T512，仅新增sigma=.5的12cell、复用新sigma0/1；检验至少3/4 meanD(.5)>=.01且各>=2/3 init>0，先明确预测/pinned源码/引用hashcommit，匹配后训练。不预测线性/二次缩放或连续阈值；全development，data同时改变输入/clean目标/归一化，不拆机制比例、跨data/noise概率或train-only早停。'
    assert state['round']==before_state['round']
    assert all(state['directions'][d]['rounds_done']==before_state['directions'][d]['rounds_done'] for d in state['directions'])
    for d in state['directions']:
        if d!=direction: assert state['directions'][d]==before_state['directions'][d]
    dump(state_path,state)
    progress=REPO/'reports/PROGRESS.md'
    with progress.open('a') as file:
        file.write(f'\n2026-10-07 · r047 · D7（方向第4轮）：固定noise733123，仅换data7339后原放大判据是否保持？5aa2a41预注册后完成24/24新cell、48轨迹、{summary["training_seconds"]:.6f}秒；P1 supported4/4、12/12 D>0，四meanD=.469371/.519513/.111892/.186296，radial×8初始化CI跨零、四跨data配对差CI均跨零。新radial×32 init47 noisy H=.036880344/C=−.021062710的分歧与旧反例保留；120历史/48新文件hash/mtime与60cell重建通过，独立解析区间首容差失败保留且仅核验改容差，新增2条有界measured claims。证据：[finding](../directions/D7_train_gain_test_harm/findings/r047_data_seed_transfer.md)、[summary](../directions/D7_train_gain_test_harm/studies/r047_data_seed_transfer/summary.json)。\n')
    dump(STUDY/'executed/central_update_audit.json',{'status':'passed','only_direction_fields_changed':['D7_train_gain_test_harm.last_round','D7_train_gain_test_harm.next_question'],'round_unchanged':state['round'],'rounds_done_unchanged':state['directions'][direction]['rounds_done'],'other_directions_unchanged':True,'new_claims':['D7-005','D7-006'],'evidence_exists':all((REPO/p).exists() for p in evidence)})
    print('finding, cumulative report, two measured claims, direction-only state, and PROGRESS saved')


if __name__=='__main__':
    main()
