/* Authored teaching sketches: examples only, never measured results. */
(() => {
  const blue = '#2855e8', ink = '#252622', line = '#d6d5c9', yellow = '#f5d96e';
  const text = (x, y, label, fill = ink, size = 20, anchor = 'middle') => `<text x="${x}" y="${y}" fill="${fill}" font-size="${size}" text-anchor="${anchor}">${label}</text>`;
  const path = (d, color = line, width = 2, extra = '') => `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" ${extra}/>`;
  const box = (x, y, w, h, fill = '#fffdf8', stroke = line, r = 5) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="2"/>`;
  const dot = (x, y, r = 5, fill = blue) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}"/>`;
  const arrow = (x1, y, x2, color = ink) => path(`M${x1} ${y}H${x2}m-7 -5 7 5-7 5`, color);
  const check = (x, y, color = blue) => path(`M${x-9} ${y}l6 6 13-15`, color, 3);
  const svg = body => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 190" fill="none" font-family="-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif" aria-hidden="true">${body}</svg>`;
  const dots = points => points.map(([x, y]) => dot(x, y)).join('');
  const sheet = (x, y, fill = '#fffdf8') => box(x, y, 68, 84, fill, ink, 3) + path(`M${x+15} ${y+21}h38M${x+15} ${y+34}h25`, line);
  window.AIQ_DIAGRAMS = {
    data: {
      alt: '同一个模型，面对规律清楚的数据和夹杂误差的数据，表现可能不同。',
      svg: svg(
        text(280, 25, '同一个模型，换一套题？') +
        box(50, 43, 190, 106) + box(320, 43, 190, 106) +
        path('M74 127H216M74 127V62') + path('M344 127H486M344 127V62') +
        path('M83 119L207 68', line, 2, 'stroke-dasharray="4 5"') +
        path('M353 119L477 68', line, 2, 'stroke-dasharray="4 5"') +
        dots([[88,117],[110,109],[134,97],[158,88],[180,79],[204,69]]) +
        dots([[358,103],[379,120],[401,72],[426,101],[450,66],[477,86]]) +
        text(145, 177, '规律清楚') + text(415, 177, '夹杂误差') + text(280, 109, '?', blue, 32)
      )
    },
    model: {
      alt: '两层和四层的模型都能接收输入、给出答案；多加几层是否有用，需要比较。',
      svg: svg(
        text(145, 29, '少几层') + text(415, 29, '多几层') +
        box(99, 67, 92, 28, '#eef1ff', blue) + box(99, 104, 92, 28, '#eef1ff', blue) +
        [45,73,101,129].map(y => box(369, y, 92, 21, '#eef1ff', blue)).join('') +
        path('M145 95V104M415 66V73M415 94V101M415 122V129', blue) +
        arrow(53, 99, 83) + arrow(207, 99, 237) + arrow(323, 99, 353) + arrow(477, 99, 507) +
        text(145, 177, '2 层') + text(415, 177, '4 层') + text(280, 106, '?', blue, 32)
      )
    },
    training: {
      alt: '示意曲线中，方案甲开始进步快，方案乙后来追上并超过它。横轴是训练步数，向上表示表现更好。',
      svg: svg(
        path('M68 146H496m-7 -5 7 5-7 5M68 146V33m-5 7 5-7 5 7', ink) +
        text(68, 23, '表现 ↑', ink, 20, 'start') + text(493, 178, '训练步数', ink, 20, 'end') +
        path('M87 134C128 69 167 55 226 61S339 77 464 64', blue, 4) +
        path('M87 137C163 133 210 116 254 91S356 40 464 34', ink, 4, 'stroke-dasharray="9 6"') +
        text(484, 68, '甲', blue) + text(484, 37, '乙') +
        text(131, 174, '早期', '#6b6c63', 20) + text(332, 174, '后期', '#6b6c63', 20)
      )
    },
    metric: {
      alt: '同一次比较，练习题上甲领先，新题上乙领先；先说明用哪类题的表现选赢家。',
      svg: svg(
        text(270, 28, '练习题') + text(430, 28, '新题') +
        box(185, 44, 320, 122) + path('M345 44V166M185 105H505') +
        text(121, 83, '甲') + text(121, 144, '乙') +
        box(199, 55, 132, 39, '#fff4be', '#fff4be') + box(359, 116, 132, 39, '#fff4be', '#fff4be') +
        text(265, 82, '领先', blue) + text(425, 143, '领先', blue) +
        text(425, 83, '—', '#6b6c63') + text(265, 144, '—', '#6b6c63')
      )
    },
    budget: {
      alt: '每批三十二个样本、更新一百次，与每批六十四个样本、更新五十次，都处理三千二百个样本；更新次数却不同。',
      svg: svg(
        text(280, 25, '每批样本 × 更新次数') +
        [0,1,2,3].map(i => box(69+i*20, 58, 13, 25, '#eef1ff', blue, 2)).join('') +
        [0,1,2,3,4,5,6,7].map(i => box(338+i*15, 58, 10, 25, '#eef1ff', blue, 2)).join('') +
        text(145, 117, '32 × 100', blue, 26) + text(415, 117, '64 × 50', blue, 26) +
        text(280, 116, '=', ink, 26) + path('M145 132v15h270v-15') +
        box(180, 146, 200, 35, '#fff4be', '#fff4be') + text(280, 171, '都处理 3200 个', ink, 20)
      )
    },
    evaluation: {
      alt: '左边重考已经见过的旧题，右边用没见过的新题验证；两者不能当作同一种证据。',
      svg: svg(
        sheet(70, 38) + sheet(156, 38) +
        dot(104, 97, 10) + dot(190, 97, 10) + arrow(141, 78, 151) +
        sheet(336, 38) + sheet(422, 38, '#fff8d9') +
        dot(370, 97, 10) + path('M445 108l11-21 11 21Z', blue, 2) + arrow(407, 78, 417) +
        text(147, 159, '旧题再考') + text(413, 159, '换未见新题') +
        path('M280 41V135', line, 2, 'stroke-dasharray="4 5"')
      )
    },
    reasoning: {
      alt: '先猜一个原因，再只改变相关的一个条件，最后看结果是否符合预测。',
      svg: svg(
        box(45, 34, 116, 96) + box(222, 34, 116, 96) + box(399, 34, 116, 96) +
        text(103, 101, '?', blue, 49) +
        path('M242 61H318M242 84H318M242 107H318', line, 3) +
        dot(262, 61, 8, '#fffdf8') + dot(298, 84, 8, yellow) + dot(270, 107, 8, '#fffdf8') +
        `<circle cx="262" cy="61" r="8" stroke="${ink}" stroke-width="2"/><circle cx="298" cy="84" r="8" stroke="${blue}" stroke-width="2"/><circle cx="270" cy="107" r="8" stroke="${ink}" stroke-width="2"/>` +
        path('M425 106V54M425 106H491', line) + path('M435 96L452 85 467 67 487 57', blue, 3) +
        arrow(177, 81, 207) + arrow(354, 81, 384) +
        text(103, 166, '猜原因') + text(280, 166, '只改一处') + text(457, 166, '看结果')
      )
    },
    knowledge: {
      alt: '经验卡应带上适用条件。遇到新题，条件匹配时参考，不匹配时先别照搬。',
      svg: svg(
        box(43, 42, 158, 110) + text(122, 75, '经验卡', blue) +
        box(58, 92, 128, 39, '#fff4be', '#fff4be') + text(122, 119, '适用条件') +
        path('M217 97H263M263 97V51H304m-7 -5 7 5-7 5M263 97V140H304m-7 -5 7 5-7 5', ink) +
        box(320, 22, 196, 58, '#eef1ff') + box(320, 111, 196, 58) +
        text(395, 59, '条件匹配') + check(481, 51) +
        text(395, 148, '先别照搬') + text(481, 150, '?', blue, 26)
      )
    }
  };
})();
