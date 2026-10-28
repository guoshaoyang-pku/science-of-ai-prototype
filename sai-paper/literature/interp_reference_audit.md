# 可解释性文献：基于一篇综述参考文献的覆盖抽查

初次检查日期及补录后复查日期：2026-10-28。抽查源为 Sharkey 等，[Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496)，**arXiv v1**，本地原文 [2501.16496.pdf](../papers/interp/2501.16496.pdf)，参考文献 PDF 第 37–40 页。该审计检查文献目录是否存在记录；原文内容、K/L/S 分类及审读深度以各条 JSON 记录为准。

## 选择和核对方法

从该综述按作者排序的参考文献开头数页中，**人工挑选**与解释性方法、表征探针、自动解释、特征字典及其评价直接相关的 17 项，其中包括几篇方法综述与近期工作；这不是随机抽样，也不是对整个参考文献列表的穷尽。它只能提示具体遗漏，不能估计库对全部经典工作或领域的覆盖率；被选入综述也不自动证明一篇论文是经典。

核对范围是本项目当前的 [interp.json](interp.json)、[rl.json](rl.json)、[pretrain.json](pretrain.json) 的 **papers 记录**。以题名（忽略大小写）、原文列出的作者与唯一来源标识为依据；arXiv 版与会议版可以是同一论文，方法名称相近或引用另一个相关版本则不足以算命中。对已定位 arXiv ID 的遗漏还搜索该 ID；对不完全一致的标题做人工检查。下表中的“未见记录”是针对检查时这三个 JSON 的判断，不排除存在未检索出的别名，也不代表外部文献不存在。

## 初次抽查结果（补录前，保留原状态）

| 综述参考文献，括号内为 PDF 页 | 主题及一手来源 | 初次抽查结果（补录前） |
|---|---|---|
| Abid 等，Meaningfully Debugging Model Mistakes using Conceptual Counterfactual Explanations (37) | [ICML 2022 / PMLR](https://proceedings.mlr.press/v162/abid22a.html)，概念反事实 | 未见记录；与已收录的 Diverse Counterfactual Explanations 是不同工作 |
| Adebayo 等，Sanity Checks for Saliency Maps (37) | [NeurIPS 2018](https://proceedings.neurips.cc/paper_files/paper/2018/file/294a8ed24b1ad22ec2e7efea049b8737-Paper.pdf)，显著图随机化检查 | **interp 已收录**；[arXiv 1810.03292](https://arxiv.org/abs/1810.03292) |
| Adebayo 等，Debugging Tests for Model Explanations (37) | [NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/075b051ec3d22dac7b33f788da631fd4-Paper.pdf)，解释调试 | **interp 已收录**；[arXiv 2011.05429](https://arxiv.org/abs/2011.05429) |
| Adler 与 Shavit，On the Complexity of Neural Computation in Superposition (37) | [arXiv 2409.15318](https://arxiv.org/abs/2409.15318)，superposition 计算复杂度 | 未见记录 |
| Agarwal 等，Faithfulness vs. Plausibility: On the (Un)Reliability of Explanations from Large Language Models (37) | [arXiv 2402.04614](https://arxiv.org/abs/2402.04614)，解释忠实性和表面合理性 | 未见记录 |
| Agarwal 等，Neural Additive Models: Interpretable Machine Learning with Neural Nets (37) | [NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/251bd0442dfcc53b5a761e050f8022b8-Paper.pdf)，可解释加性模型 | 未见记录 |
| Alain 与 Bengio，Understanding Intermediate Layers Using Linear Classifier Probes (37) | [OpenReview](https://openreview.net/forum?id=ryF7rTqgl)，线性探针；另有 [arXiv 1610.01644](https://arxiv.org/abs/1610.01644) | 未见记录；已有其他 probe 文献不能代替该来源 |
| Bau 等，Understanding the Role of Individual Units in a Deep Neural Network (38) | [PNAS，doi:10.1073/pnas.1907375117](https://doi.org/10.1073/pnas.1907375117)；另有 [arXiv 2009.05041](https://arxiv.org/abs/2009.05041) | 未见记录；已有的 2017 Network Dissection 与此 2020 论文相关但不是同一条文献 |
| Belinkov，Probing Classifiers: Promises, Shortcomings, and Advances (38–39) | [Computational Linguistics 2022](https://aclanthology.org/2022.cl-1.7/)，探针综述 | **interp 已收录**；[arXiv 2102.12452](https://arxiv.org/abs/2102.12452)。综述中 Belinkov 2022a/b 为重复引文，此处只计一次 |
| Belrose 等，Eliciting Latent Predictions from Transformers with the Tuned Lens (39) | [arXiv 2303.08112](https://arxiv.org/abs/2303.08112)，层间读出 | **interp 已收录** |
| Belrose 等，LEACE: Perfect Linear Concept Erasure in Closed Form (39) | [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/d066d21c619d0a78c5b557fa3291a8f4-Paper-Conference.pdf)，线性概念擦除 | **interp 已收录**；[arXiv 2306.03819](https://arxiv.org/abs/2306.03819) |
| Bereska 与 Gavves，Mechanistic Interpretability for AI Safety — A Review (39) | [TMLR](https://openreview.net/forum?id=ePUVetPKu6)，综述 | 未见记录 |
| Bills 等，Language Models Can Explain Neurons in Language Models (39) | [作者原生文章](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)，自动神经元解释 | 未见记录；已有多模态特征自动解释论文与之不同 |
| Bolukbasi 等，An Interpretability Illusion for BERT (39) | [arXiv 2104.07143](https://arxiv.org/abs/2104.07143)，解释错觉 | 未见记录；与已收录的 subspace patching illusion 是不同文章 |
| Borowski 等，Exemplary Natural Images Explain CNN Activations Better Than State-of-the-Art Feature Visualization (39) | [ICLR 2021](https://openreview.net/forum?id=QO9-y8also-)，自然图像与特征可视化 | 未见记录 |
| Braun 等，Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning (39) | [NeurIPS 2024 / OpenReview](https://openreview.net/forum?id=7txPaUpUnc)；另有 [arXiv 2405.12241](https://arxiv.org/abs/2405.12241)，端到端字典 | 未见记录；现有 SAE 论文和概念不等于收录该论文 |
| Brendel 与 Bethge，Approximating CNNs with Bag-of-Local-Features Models Works Surprisingly Well on ImageNet (40) | [ICLR 2019](https://openreview.net/forum?id=SkfMWhAqYQ)，BagNet | **interp 已收录**；[arXiv 1904.00760](https://arxiv.org/abs/1904.00760) |

初次抽查时，这 17 个方便抽选项中，6 项在 interp 目录有对应记录，11 项在三个领域的论文记录中均未发现对应条目；**本样本中没有“仅收录在 RL 或 pretrain”的跨领域命中**。

## 补录后复查

上述 11 项现已补入 [interp.json](interp.json)，本样本因此为 **17 项在 interp 有记录、0 项未见记录**；补录前后的表述只涉及这组方便抽查。新补 10 份 PDF 均经首页题名与 arXiv 编号核验，OpenAI 文章保留作者原生 HTML、含正文的 JavaScript bundle 及模板 JS；其他资源和离线交互未全部验证。各项均仅核对了指定部分，未宣称全文阅读或实验复现。Adler–Shavit 的一条理论命题在原文限定下存在精确量词反例，见 [interp_coverage.md](interp_coverage.md) 的本轮记录；“已收录”不等于论文全部结果正确。

对其余综述参考文献及三个领域以外的记录没有作此结论，更不能将本样本的前后数量外推为领域总体覆盖率。

## 2026-10-28 后续经典源头补录

对 [interp_coverage.md](interp_coverage.md) 中此前列为待核的10项继续回溯原文，现已补入：RLACE、Shapley1953原章的1988正式重印、Smolensky1986 Volume 2 Chapter22、Szegedy Intriguing Properties、2022 Taking Features Out of Superposition、Bilinear技术笔记、MAIA、电路假说统计检验、AlphaZero概念发现与迁移、Strong Feature Hypothesis作者论述。此处仅说明这10项已建立材料和具体主张记录；材料类型、版本、定向审读位置、公式或版本争议见该覆盖文档及interp.json。

其中2项主张列为有条件Science（Shapley公理唯一性、双线性层张量恒等式），8项列为Knowledge（方法、有限观察或方法论论证），新增10个相应概念。该分类不表示对整篇其余内容完成审核。原始抽查表保留历史状态；pp.62–67的对应候选行更新为“后续已补”，其余待核项和后续未读参考页保持开放。没有据此估计总体覆盖率或宣布领域经典已穷尽。

## 2026-10-28 后续第二批补录

继续闭合此前pp.56–67队列的10项：Mamba事实定位编辑、稀疏SAE连接、Current Themes、Coup Probes、Mechanistic?、Early Vision、MIPS、RNN迁移、双线性权重分解、BIMT。均已回到一手正文并记录指定页码／段落、实际版本与分类边界；不是仅凭综述题名分类。BIMT旧待核备注把它误归为自动特征解释，现按Brain-Inspired Modular Training原文纠正；RNN和双线性论文以原文题名登记、保留综述别名。

本批6份主PDF、4份原生HTML、10个具体概念入库，interp为309项／296概念。仅双线性二次型分解的条件代数列Science，应用解释和其余方法／讨论列Knowledge；没有新增Law。已更新对应10行状态，其他待核和pp.68以后的未读引用继续开放。Roweis–Ghahramani线性Gaussian统一综述仅完成作者源与摘要定位，未计为正文已审或纳入条目。

## 2026-10-28 后续第三批补录与书目终点

本批新增8篇论文/综述、6个独立概念，interp为317项/302概念（299主PDF、18主HTML）。6篇来自pp.62–65旧队列：Selective Pruning、Fine-tuning Enhances Existing Mechanisms、Practical Review、Category-free Neural Population、Toward Transparent AI、Beyond Model Interpretability；2篇来自p.74：Yun 2021共享字典与Zhou 2015对象检测器。全部新增主张为Knowledge，2篇综述复用既有概念。作者原生指定评论Feature Is Overloaded Terminology已核正文，作为补充source注释，未计独立paper。

实际版本、页码与外延见interp.json及覆盖文档第三批表。特别保留：(1) Selective Pruning v2 p.3 Definition 2与Algorithm 1比值方向不一致，已目检，但实现未审；(2) Practical Review请求v1实际返回v4（2025-10-13、61页），按实际正文登记，不冒称2024初稿；(3) Raposo为HHS/PMC作者稿，页码依33页作者稿，不冒称期刊排版页。

本批以单次8页提取推进2501.16496v1 pp.68–75：Gemini题名在p.69落定，现已跨库补入pretrain并核对官方修订稿指定章节；p.74为书目终点，p.75为附录。对pp.70–74手工选择44项核对三库身份：2项本批补、12项既有、30项待核。该选择不覆盖这些页所有跨域/政策/历史条目，更不是44项原文全部已审。旧pp.62–67的59行当前为8原批纳入、20既有、22后续已补（含pretrain的Gemini）、1指定评论原文已核、8待核。pp.56–61原有7项待核仍开放；登记队列、暂缓归属及找到书目终点均不计全文覆盖完成。
