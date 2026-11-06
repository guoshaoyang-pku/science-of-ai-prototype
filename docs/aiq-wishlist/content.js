window.AIQ_ASPECTS = [
  {
    id: "data",
    guide: {
      "scope": "研究给模型学什么：有多少例子，数据里有没有干扰。",
      "importance": "同一套训练方案，遇到不同的题，不一定都合适。",
      "scenario": "同样的模型，分别学清晰的数据和带干扰的数据：原来的训练方法还合适吗？",
      "reading": {
        "id": "D2",
        "text": "D2 原先根据数据量和干扰大小预测何时停止训练，扩大比较范围后失败。少数题上的经验，不能直接套用到所有题。"
      },
      "prompt": "我希望 AIQ 能判断数据变得不准后，是否需要换训练方案。"
    },
    exampleIds: ["D2", "D3", "D6", "D7", "D11"],
    title: "数据与任务",
    question: "题目变了，还选同一个方案吗？",
    explanation: "先看给模型什么题：数据多不多、准不准，都可能改变选择。",
    current: "AIQ 已有让赢家随数据变化的题目设计；哪些数据特征能帮助稳定预测，仍待检验。",
    wish: "我希望 AIQ 能判断数据变得不准后，是否需要换训练方案。",
    next: "先选一种数据，只改变一个因素，写清希望预测的变化。",
    placeholder: "例如：数据里多了一些干扰，还该用原来的方法吗？",
    sourceLabel: "查看完整说明与依据",
    sourceHref: "https://github.com/guoshaoyang-pku/science-of-ai-prototype/blob/main/docs/AIQ_WISHLIST.md#数据与任务"
  },
  {
    id: "model",
    guide: {
      "scope": "研究模型怎么搭：搭几层，每层放多少部件。",
      "importance": "模型更复杂，不代表在同样的训练预算下就表现更好。",
      "scenario": "同一批题、同样的预算，一个模型搭两层，另一个搭四层：多出的两层值得吗？",
      "reading": {
        "id": "D5",
        "text": "D5 只在几种任务和特定指标上看到加一个结构部件的改善，原因还没解释清楚，不能推成普遍建议。"
      },
      "prompt": "我希望 AIQ 能告诉我，什么时候值得把模型搭得更复杂。"
    },
    exampleIds: ["D3", "D5", "D6", "D8", "D10", "D11", "D13"],
    title: "模型结构",
    question: "模型搭得更复杂，就更好吗？",
    explanation: "像搭积木一样，模型可以多搭几层、每层放更多部件。",
    current: "已有研究记录了结构收益与反例，但局部现象还不能变成普遍的结构建议。",
    wish: "我希望 AIQ 能告诉我，什么时候值得把模型搭得更复杂。",
    next: "先选一个结构改动，列出已知收益、反例和比较条件。",
    placeholder: "例如：同样的训练预算，多加两层会更好吗？",
    sourceLabel: "查看完整说明与依据",
    sourceHref: "https://github.com/guoshaoyang-pku/science-of-ai-prototype/blob/main/docs/AIQ_WISHLIST.md#模型结构"
  },
  {
    id: "training",
    guide: {
      "scope": "研究模型怎么学：每次改多少，学多少轮，怎样保持稳定。",
      "importance": "眼前领先，不代表到约定的训练终点仍然领先。",
      "scenario": "假设允许看开头一段训练：甲起步快，乙后来追上。AIQ 能提前判断谁最后更好吗？",
      "reading": {
        "id": "D4",
        "text": "D4 在简化任务中发现，一种曲线画法不能描述所有训练过程。看完整过程能画得像，也不等于只看开头就能猜中。"
      },
      "prompt": "我希望在允许看早期训练时，AIQ 能判断谁会后来居上。"
    },
    exampleIds: ["D1", "D4", "D5", "D6", "D9", "D10", "D12", "D13"],
    title: "优化与训练",
    question: "起步快的，最后也会赢吗？",
    explanation: "模型相同，学习的方法不同，也可能一个先快、一个后来居上。",
    current: "已有训练曲线研究和反例；允许观察早期轨迹的预测，需要另定扩展评测协议。",
    wish: "我希望在允许看早期训练时，AIQ 能判断谁会后来居上。",
    next: "先约定能看多少步，以及要预测哪个训练终点。",
    placeholder: "例如：如果允许看开头一段训练，能猜中最后的赢家吗？",
    sourceLabel: "查看完整说明与依据",
    sourceHref: "https://github.com/guoshaoyang-pku/science-of-ai-prototype/blob/main/docs/AIQ_WISHLIST.md#优化与训练动力学"
  },
  {
    id: "metric",
    guide: {
      "scope": "研究用什么标准选赢家，先把“好”说清楚。",
      "importance": "比较标准不同，同一组方案也可能选出不同赢家。",
      "scenario": "甲在练习题上更好，乙在新题上更好。如果我们想选会做新题的方案，就该看后者。",
      "reading": {
        "id": "D7",
        "text": "D7 发现，练习题上有优势，新题上却未必。它还没有找到只看训练过程就能可靠决定何时停止的方法。"
      },
      "prompt": "我希望 AIQ 能分清练习题上的进步和新题上的进步。"
    },
    exampleIds: ["D2", "D7", "D9"],
    title: "目标与评价",
    question: "练习题做得好，新题也会吗？",
    explanation: "先说清楚什么算好：是做对练过的题，还是做好没见过的题？",
    current: "固定特征研究发现，风险回升不要求训练误差先很低；仅凭训练信息早停仍未建立。",
    wish: "我希望 AIQ 能分清练习题上的进步和新题上的进步。",
    next: "先列出决策时能看到的信息，再定义早停好坏的指标。",
    placeholder: "例如：练习成绩更好，为什么新题反而做得更差？",
    sourceLabel: "查看完整说明与依据",
    sourceHref: "https://github.com/guoshaoyang-pku/science-of-ai-prototype/blob/main/docs/AIQ_WISHLIST.md#目标与评价指标"
  },
  {
    id: "budget",
    guide: {
      "scope": "研究有限的训练资源怎么分配，先说清用什么单位计算。",
      "importance": "处理相同数量的数据，不代表花了相同时间或算力。",
      "scenario": "同样处理 3200 个样本：每次 32 个学 100 次，或每次 64 个学 50 次，该选哪种？",
      "reading": {
        "id": "D1",
        "text": "D1 尝试换算不同方案的训练进度，但换算后的时间不能直接代表学得多好。早期有用，也不能保证后期有用。"
      },
      "prompt": "我希望 AIQ 能在相同预算下，帮我选合适的训练安排。"
    },
    exampleIds: ["D1", "D9", "D11"],
    title: "资源与预算",
    question: "资源有限，怎么分配更划算？",
    explanation: "先约定能花多少资源，再比较该一次多学点，还是多学几次。",
    current: "AIQ 已有明确预算和跨预算比较协议；具体采用哪种资源口径，需要按问题确定。",
    wish: "我希望 AIQ 能在相同预算下，帮我选合适的训练安排。",
    next: "先选定预算单位，再比较低预算和高预算下的选择。",
    placeholder: "例如：每次多学点和多学几次，哪种更划算？",
    sourceLabel: "查看完整说明与依据",
    sourceHref: "https://github.com/guoshaoyang-pku/science-of-ai-prototype/blob/main/docs/AIQ_WISHLIST.md#资源与公平比较"
  },
  {
    id: "evaluation",
    guide: {
      "scope": "研究怎样确认进步可信：题目是否见过，比较是否公平。",
      "importance": "反复做同一套题得高分，不能直接说明会做新题。",
      "scenario": "看过答案后，重做旧题得高分还不够。先固定方法，再拿一套没见过的题检验。",
      "reading": {
        "id": "D6",
        "text": "D6 的预测方法在原来抽到的数据上有帮助，换一批数据后却没能稳定保留效果；这也不是最终做题成绩的检验。"
      },
      "prompt": "我希望 AIQ 的每次进步，都能用没见过的题再检验。"
    },
    exampleNote: "从评测方法看已有问题（建议关联）：怎样验证、何时失效？",
    exampleIds: ["D4", "D6", "D13"],
    title: "评测可信度",
    question: "是真学会了，还是记住答案了？",
    explanation: "高分还要经得起检查：换一套没见过的题，还能做得好吗？",
    current: "已有生成代码执行、输入冻结等协议；每项提升仍需对应清晰的独立条件和对照。",
    wish: "我希望 AIQ 的每次进步，都能用没见过的题再检验。",
    next: "先写一页评测约定：测什么、与谁比、哪些信息可见。",
    placeholder: "例如：怎样排除“只是记住了答案”这种可能？",
    sourceLabel: "查看完整说明与依据",
    sourceHref: "https://github.com/guoshaoyang-pku/science-of-ai-prototype/blob/main/docs/AIQ_WISHLIST.md#答案与评测可信度"
  },
  {
    id: "reasoning",
    guide: {
      "scope": "研究一个解释能否接受检验，以及怎样发现它错了。",
      "importance": "答案猜对了，理由也可能不对；需要再设计比较来检查。",
      "scenario": "猜测“每次学得太猛会不稳定”：只把每次调整的幅度调小，先预测变化，再实际比较。",
      "reading": {
        "id": "D2",
        "text": "D2 原先关于停止时间的普遍预测被推翻，但简化条件下的数学关系仍成立。公式正确，不代表据此猜出的规律也正确。"
      },
      "prompt": "我希望 AIQ 给出解释时，也告诉我怎么验证它。"
    },
    exampleIds: ["D2", "D4", "D5", "D8", "D12", "D13"],
    title: "推理与解释",
    question: "这个解释，能试出来吗？",
    explanation: "不只说“为什么”，还要说出：改变一个条件，会发生什么。",
    current: "已有可检验问题与预注册流程；解析关系和局部经验仍需分别验证，不能直接外推。",
    wish: "我希望 AIQ 给出解释时，也告诉我怎么验证它。",
    next: "找两种竞争解释，写出一个能让它们预测不同的条件。",
    placeholder: "例如：只改变一个条件，怎样检验这个解释？",
    sourceLabel: "查看完整说明与依据",
    sourceHref: "https://github.com/guoshaoyang-pku/science-of-ai-prototype/blob/main/docs/AIQ_WISHLIST.md#推理与解释"
  },
  {
    id: "knowledge",
    guide: {
      "scope": "研究怎样保存有依据的经验，并判断新任务是否适用。",
      "importance": "记下很多经验不等于更会解题，还要检验用后是否更好。",
      "scenario": "一张经验卡写着“只在输入数值较小时适用”。遇到更大的数值，就先别照搬。",
      "reading": {
        "id": "D8",
        "text": "D8 在简化网络、输入数值较小时找到规律，换成较大数值后失效。这说明经验要带条件，尚未证明能帮助 AIQ 解题。"
      },
      "prompt": "我希望 AIQ 能借用可靠经验，也提醒我什么时候不适用。"
    },
    exampleNote: "从知识外推看已有问题（建议关联）；不代表已证明解题收益。",
    exampleIds: ["D2", "D8", "D11"],
    title: "知识与迁移",
    question: "以前的经验，新题还能用吗？",
    explanation: "给经验贴上使用条件，遇到新问题时，先看能不能用。",
    current: "已有知识冻结、修订和谱系管理；本仓库新协议下的解题收益尚待独立验证。",
    wish: "我希望 AIQ 能借用可靠经验，也提醒我什么时候不适用。",
    next: "先挑选范围清楚的知识，设计有知识与无知识的对照。",
    placeholder: "例如：遇到新任务，哪些旧经验可以用，哪些不能？",
    sourceLabel: "查看完整说明与依据",
    sourceHref: "https://github.com/guoshaoyang-pku/science-of-ai-prototype/blob/main/docs/AIQ_WISHLIST.md#知识积累与迁移"
  }
];
