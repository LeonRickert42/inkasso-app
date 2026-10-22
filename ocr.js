// Texterkennung im Browser: PaddleOCR (Detektor PP-OCRv4 + lateinische Erkennung PP-OCRv3) via onnxruntime-web.
// Läuft komplett auf dem Gerät. Liefert Zeilen [{y, x, h, text}] in Pixeln, sortiert von oben nach unten.
/* global ort */
const OCR = (() => {
  let det, rec, chars, loading;
  const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

  function init(progress) {
    if (!loading) loading = load(progress).catch(e => { loading = null; throw e; });
    else progress?.('Lade Texterkennung …');
    return loading;
  }

  async function load(progress) {
    ort.env.wasm.wasmPaths = new URL('vendor/', location.href).href;
    ort.env.wasm.numThreads = 1;
    const load = async (f, label) => {
      progress?.(label);
      const r = await fetch('vendor/' + f);
      if (!r.ok) throw new Error(f + ' nicht ladbar');
      return new Uint8Array(await r.arrayBuffer());
    };
    const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    det = await ort.InferenceSession.create(await load('det.onnx', 'Lade Texterkennung (1/2) …'), opt);
    rec = await ort.InferenceSession.create(await load('rec.onnx', 'Lade Texterkennung (2/2) …'), opt);
    const dict = (await (await fetch('vendor/dict.txt')).text()).split(/\r?\n/).filter(l => l.length);
    chars = ['', ...dict, ' '];
  }

  function canvasOf(w, h) {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    return [c, c.getContext('2d', { willReadFrequently: true })];
  }

  // Wie RapidOCR: kurze Seite mind. 736 px, lange höchstens 2000 px, Vielfaches von 32
  function detSize(w, h) {
    let r = Math.min(w, h) < 736 ? 736 / Math.min(w, h) : 1;
    if (Math.max(w, h) * r > 2000) r = 2000 / Math.max(w, h);
    return [Math.max(32, Math.round(w * r / 32) * 32), Math.max(32, Math.round(h * r / 32) * 32)];
  }

  async function detect(bmp) {
    const [W, H] = detSize(bmp.width, bmp.height);
    const [, ctx] = canvasOf(W, H);
    ctx.drawImage(bmp, 0, 0, W, H);
    const px = ctx.getImageData(0, 0, W, H).data, n = W * H;
    const input = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++)  // BGR wie OpenCV
      input[c * n + i] = (px[i * 4 + 2 - c] / 255 - MEAN[c]) / STD[c];
    const out = await det.run({ [det.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, H, W]) });
    const prob = out[det.outputNames[0]].data;

    // DB-Nachbearbeitung: Schwellwert, Zusammenhangskomponenten, gedrehtes Rechteck (Fotos sind oft schief)
    const seen = new Uint8Array(n), boxes = [], stack = [], pts = [];
    const sx = bmp.width / W, sy = bmp.height / H;
    for (let s = 0; s < n; s++) {
      if (seen[s] || prob[s] <= 0.3) continue;
      pts.length = 0;
      let mx = 0, my = 0, score = 0;
      stack.push(s); seen[s] = 1;
      while (stack.length) {
        const p = stack.pop(), x = p % W, y = (p - x) / W;
        pts.push(p); mx += x; my += y; score += prob[p];
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy, q = ny * W + nx;
          if (nx >= 0 && ny >= 0 && nx < W && ny < H && !seen[q] && prob[q] > 0.3) { seen[q] = 1; stack.push(q); }
        }
      }
      if (pts.length < 10 || score / pts.length < 0.5) continue;
      mx /= pts.length; my /= pts.length;
      let cxx = 0, cyy = 0, cxy = 0;
      for (const p of pts) { const x = p % W - mx, y = (p - p % W) / W - my; cxx += x * x; cyy += y * y; cxy += x * y; }
      const raw = 0.5 * Math.atan2(2 * cxy, cxx - cyy);
      const ang = Math.abs(raw) < 0.025 || Math.abs(raw) > 0.6 ? 0 : raw;  // fast gerade → achsparallel lassen
      const c = Math.cos(ang), si = Math.sin(ang);
      let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
      for (const p of pts) {
        const x = p % W - mx, y = (p - p % W) / W - my, u = x * c + y * si, v = -x * si + y * c;
        if (u < u0) u0 = u; if (u > u1) u1 = u; if (v < v0) v0 = v; if (v > v1) v1 = v;
      }
      let len = u1 - u0 + 1, th = v1 - v0 + 1;
      if (Math.min(len, th) < 3) continue;
      const d = len * th * 1.6 / (2 * (len + th));
      len += 2 * d; th += 2 * d;
      const ux = mx + (u0 + u1) / 2 * c - (v0 + v1) / 2 * si, uy = my + (u0 + u1) / 2 * si + (v0 + v1) / 2 * c;
      // in Bildkoordinaten umrechnen (sx ≈ sy, da Seitenverhältnis erhalten bleibt)
      const k = (sx + sy) / 2;
      const box = { cx: (ux + 0.5) * sx, cy: (uy + 0.5) * sy, len: len * k, th: th * k, ang, slope: Math.abs(raw) > 0.6 ? 0 : Math.tan(raw) };
      if (Math.min(box.len, box.th) < 5) continue;
      boxes.push(box);
    }
    return boxes;
  }

  async function recognize(bmp, b) {
    const H = 48, w = Math.max(1, Math.ceil(H * b.len / b.th)), W = Math.max(320, w);
    const [, ctx] = canvasOf(w, H);
    const k = H / b.th;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, w, H);
    ctx.translate(w / 2, H / 2); ctx.scale(k, k); ctx.rotate(-b.ang); ctx.translate(-b.cx, -b.cy);
    ctx.drawImage(bmp, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    const px = ctx.getImageData(0, 0, w, H).data;
    const input = new Float32Array(3 * H * W);  // rechts mit 0 aufgefüllt
    for (let y = 0; y < H; y++) for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      for (let c = 0; c < 3; c++) input[c * H * W + y * W + x] = px[i + 2 - c] / 127.5 - 1;
    }
    const out = await rec.run({ [rec.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, H, W]) });
    const t = out[rec.outputNames[0]], [, T, C] = t.dims, p = t.data;
    let text = '', last = 0;
    for (let i = 0; i < T; i++) {
      let best = 0, bv = -Infinity;
      for (let c = 0; c < C; c++) if (p[i * C + c] > bv) { bv = p[i * C + c]; best = c; }
      if (best && best !== last) text += chars[best] ?? '';
      last = best;
    }
    // Das latin-Modell liest „ß“ oft als „B“ (z. B. „WeiBbier“)
    return text.trim().replace(/(?<=[a-zäöü])B(?=[a-zäöü])/g, 'ß');
  }

  // Bild (File/Blob) → Textzeilen eines Bons
  async function read(file, progress) {
    await init(progress);
    let bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
    const big = Math.max(bmp.width, bmp.height);
    if (big > 2600) {  // Handyfotos (12 MP) verkleinern: schneller und spart Speicher auf dem iPhone
      const f = 2600 / big, small = await createImageBitmap(bmp, { resizeWidth: Math.round(bmp.width * f), resizeHeight: Math.round(bmp.height * f), resizeQuality: 'high' });
      bmp.close?.(); bmp = small;
    }
    progress?.('Suche Text …');
    const boxes = (await detect(bmp)).filter(b => b.th < b.len * 1.5);
    const obs = [];
    for (let i = 0; i < boxes.length; i++) {
      if (i % 8 === 0) progress?.(`Lese Text … ${Math.round(i / boxes.length * 100)} %`);
      const b = boxes[i], text = await recognize(bmp, b);
      if (text) obs.push({ y: b.cy, x: b.cx - b.len / 2, cx: b.cx, h: b.th, long: b.len > 4 * b.th, slope: b.slope, text });
    }
    const mid = bmp.width / 2;
    bmp.close?.();
    // Zu Bon-Zeilen gruppieren. Bei schiefen/gewölbten Fotos liegt der Preis rechts höher oder tiefer
    // als der Name links → jede Box mit der lokalen Textneigung auf eine gemeinsame x-Position projizieren.
    const long = obs.filter(o => o.long);
    const localSlope = y => {
      const sl = long.slice().sort((a, b) => Math.abs(a.y - y) - Math.abs(b.y - y)).slice(0, 5).map(o => o.slope).sort((a, b) => a - b);
      return sl.length ? sl[sl.length >> 1] : 0;
    };
    for (const o of obs) o.py = o.y - localSlope(o.y) * (o.cx - mid);
    obs.sort((a, b) => a.py - b.py);
    const rows = [];
    let cur = [], sum = 0;
    for (const o of obs) {
      if (cur.length && Math.abs(o.py - sum / cur.length) > 0.45 * Math.min(o.h, ...cur.map(c => c.h))) { rows.push(cur); cur = []; sum = 0; }
      cur.push(o); sum += o.py;
    }
    if (cur.length) rows.push(cur);
    return rows.map(r => r.sort((a, b) => a.x - b.x).map(o => o.text).join(' '));
  }

  return { init, read };
})();
