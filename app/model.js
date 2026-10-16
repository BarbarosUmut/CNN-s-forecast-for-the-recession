// TabularCNN — tarayıcıda çıkarım (PyTorch modelinin birebir karşılığı, BatchNorm katlanmış)
function createModel(M) {
  const P = M.prep, A = M.agirliklar;
  const NUM = P.num_cols, CAT = P.cat_cols, NN = NUM.length, NT = NUM.length + CAT.length;
  const D = A.num_w[0].length;
  const logSet = new Set(P.log_cols);
  const order = M.token_sirasi;
  const flat = (a) => Float64Array.from(a.flat(Infinity));
  const conv = (c) => ({ W: flat(c.W), b: Float64Array.from(c.b), out: c.W.length, inp: c.W[0].length });
  const C1 = conv(A.conv1), C2 = conv(A.conv2), C3 = conv(A.conv3);
  const F1W = flat(A.fc1.W), F1b = Float64Array.from(A.fc1.b), F2W = flat(A.fc2.W), F2b = A.fc2.b[0];
  const vocabIdx = CAT.map((c) => new Map(P.vocab[c].map((v, i) => [v, i + 1])));

  // Conv1d(kernel=3, padding=1) + ReLU; x: Float64Array [inp * L]
  function convRelu(c, x, L) {
    const y = new Float64Array(c.out * L);
    for (let o = 0; o < c.out; o++) {
      for (let l = 0; l < L; l++) {
        let s = c.b[o];
        for (let i = 0; i < c.inp; i++) {
          const wb = (o * c.inp + i) * 3, xb = i * L;
          if (l > 0) s += c.W[wb] * x[xb + l - 1];
          s += c.W[wb + 1] * x[xb + l];
          if (l < L - 1) s += c.W[wb + 2] * x[xb + l + 1];
        }
        y[o * L + l] = s > 0 ? s : 0;
      }
    }
    return y;
  }

  // Ham müşteri kaydı -> kalibre temerrüt olasılığı
  function predict(rec) {
    const tokens = new Array(NT);
    for (let j = 0; j < NN; j++) {
      let v = Number(rec[NUM[j]]);
      if (logSet.has(NUM[j])) v = Math.log1p(Math.max(v, 0));
      v = (v - P.mean[j]) / P.std[j];
      v = Math.max(-P.clip, Math.min(P.clip, v));
      const t = new Float64Array(D);
      for (let k = 0; k < D; k++) t[k] = v * A.num_w[j][k] + A.num_b[j][k];
      tokens[j] = t;
    }
    for (let j = 0; j < CAT.length; j++) {
      const idx = vocabIdx[j].get(rec[CAT[j]]) || 0;
      tokens[NN + j] = A.embs[j][idx];
    }
    const L = NT, x = new Float64Array(D * L);
    for (let l = 0; l < L; l++) { const t = tokens[order[l]]; for (let k = 0; k < D; k++) x[k * L + l] = t[k]; }
    const h1 = convRelu(C1, x, L), h2 = convRelu(C2, h1, L), h3 = convRelu(C3, h2, L);
    const ch = C3.out, pooled = new Float64Array(2 * ch);
    for (let c = 0; c < ch; c++) {
      let s = 0, m = -Infinity;
      for (let l = 0; l < L; l++) { const v = h2[c * L + l] + h3[c * L + l]; s += v; if (v > m) m = v; }
      pooled[c] = s / L; pooled[ch + c] = m;
    }
    const hid = F1b.length; let z = F2b;
    for (let o = 0; o < hid; o++) {
      let s = F1b[o];
      for (let i = 0; i < 2 * ch; i++) s += F1W[o * 2 * ch + i] * pooled[i];
      if (s > 0) z += F2W[o] * s;
    }
    return 1 / (1 + Math.exp(-(z - M.log_pos_weight)));
  }

  // Occlusion açıklaması: her değişkeni tipik müşteri değerine çekince risk ne kadar değişiyor?
  function explain(rec) {
    const p = predict(rec);
    const out = [];
    for (const f of [...NUM, ...CAT]) {
      if (rec[f] === M.referans[f]) { out.push({ f, delta: 0, pRef: p }); continue; }
      const r = { ...rec, [f]: M.referans[f] };
      const pr = predict(r);
      out.push({ f, delta: p - pr, pRef: pr });
    }
    return { p, contrib: out };
  }
  return { predict, explain, NUM, CAT };
}
if (typeof module !== "undefined") module.exports = { createModel };
