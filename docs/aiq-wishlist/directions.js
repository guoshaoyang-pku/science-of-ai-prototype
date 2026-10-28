// Existing research questions and boundaries, linked to the readable report site.
// Reports remain historical evidence, not new findings or experiment plans.
window.AIQ_DIRECTIONS = [
  {
    id: 'D1',
    title: '等效训练时间',
    question: '不同优化器的训练进度，能换算成同一把时间尺吗？',
    boundary: '位移不等于损失进度；动量修正仍有晚期反例，自适应优化器未实测。',
    sourceHref: '../模糊方向/D1_u_effective_time.html',
    literatureHref: '../模糊方向/D1_文献深度确认.html'
  },
  {
    id: 'D2',
    title: '风险最低点',
    question: '噪声和样本量变了，风险何时回升，能提前知道吗？',
    boundary: '固定特征实验中扩域反比律失败；仅凭训练信息预测早停仍未建立。',
    sourceHref: '../模糊方向/D2_overfitting_u_curve.html',
    literatureHref: '../模糊方向/D2_文献深度确认.html'
  },
  {
    id: 'D3',
    title: '谱指标与训练步数',
    question: '初始谱指标相近的模型，会以相同速度学会目标吗？',
    boundary: '固定特征下少数标量不足以确定预算；空间分辨率尚未操作测量。',
    sourceHref: '../模糊方向/D3_capacity_resolution.html',
    literatureHref: '../模糊方向/D3_文献深度确认.html'
  },
  {
    id: 'D4',
    title: '单段训练曲线',
    question: '一条 S 形曲线能描述训练吗，何时会失效？',
    boundary: '成败依赖谱比、慢损失占比和拟合协议；换协议仍不能消除全部失败。',
    sourceHref: '../模糊方向/D4_sigmoid_training_curve.html',
    literatureHref: '../模糊方向/D4_文献深度确认.html'
  },
  {
    id: 'D5',
    title: '层归一化的收益',
    question: '宽度相同时，层归一化为什么能改善某些任务的拟合？',
    boundary: '受控任务中已测到收益，但条件数、目标对齐等候选尚未解释收益。',
    sourceHref: '../模糊方向/D5_ln_geometry.html',
    literatureHref: '../模糊方向/D5_文献深度确认.html'
  },
  {
    id: 'D6',
    title: '早期变化与晚期表现',
    question: '只看早期特征的变化，能预测最后学得怎样吗？',
    boundary: '原抽样中的预测关系未迁移到指定新抽样；影响因素尚未独立归因。',
    sourceHref: '../模糊方向/D6_kernel_drift.html',
    literatureHref: '../模糊方向/D6_文献深度确认.html'
  },
  {
    id: 'D7',
    title: '训练收益与泛化损害',
    question: '为什么训练集上更好，新数据上反而更差？',
    boundary: '终点差距、噪声扩大差距和晚期回升需分别检验；结果依赖噪声实现。',
    sourceHref: '../模糊方向/D7_train_gain_test_harm.html',
    literatureHref: '../模糊方向/D7_文献深度确认.html'
  },
  {
    id: 'D8',
    title: '激活与偏置通路',
    question: '加入偏置、换输入分布或加深后，激活通路的规律还成立吗？',
    boundary: '小尺度规律有受控证据，有限尺度已有反例；性能收益未验证。',
    sourceHref: '../模糊方向/D8_activation_pathway.html',
    literatureHref: '../模糊方向/D8_文献深度确认.html'
  },
  {
    id: 'D9',
    title: '小批量噪声',
    question: '每次少看一些样本，会怎样改变训练节奏和风险最低点？',
    boundary: '两个条件中减小批量使最低点延后；尚不能据此给出连续变化规律。',
    sourceHref: '../模糊方向/D9_minibatch_noise.html',
    literatureHref: '../模糊方向/D9_文献深度确认.html'
  },
  {
    id: 'D10',
    title: '深度中的动力学',
    question: '网络加深后，训练速度由最慢的一层决定吗？',
    boundary: '最慢参数组会高估训练时间；已测时间也不随深度单调变化。',
    sourceHref: '../模糊方向/D10_depth_propagation.html',
    literatureHref: '../模糊方向/D10_文献深度确认.html'
  },
  {
    id: 'D11',
    title: '固定特征维数与训练量',
    question: '特征维数和平均学习速率，能预测需要训练多少步吗？',
    boundary: '目前测的是固定特征维数，并非真实网络宽度；特征学习边界未测。',
    sourceHref: '../模糊方向/D11_width_scaling.html',
    literatureHref: '../模糊方向/D11_文献深度确认.html'
  },
  {
    id: 'D12',
    title: '多阶段训练曲线',
    question: '一条 S 形曲线不够时，能拆成几段并预测转折点吗？',
    boundary: '双段候选仅在部分离散条件通过；三模态、阶段合并与真跃迁未测。',
    sourceHref: '../模糊方向/D12_cascade_sigmoid.html',
    literatureHref: '../模糊方向/D12_文献深度确认.html'
  },
  {
    id: 'D13',
    title: '线性网络的有效深度',
    question: '只看线性网络的结构和初始化，能预测训练转折时间吗？',
    boundary: '无跳连时深度单调标定已失败；结果依赖读出方式，跳连组合律未测。',
    sourceHref: '../模糊方向/D13_linear_flow_world.html',
    literatureHref: '../模糊方向/D13_文献深度确认.html'
  }
];
