// Bon-Parser (Edeka-App, Lidl Plus, Papierbons z.B. Apotheke) – Gegenstück zu ocr.py/app.py der Mac-Version.
const Parse = (() => {
  const PRICE = String.raw`-?\d{1,4}[.,]\d{2}`;
  const L = String.raw`\p{L}`;
  const LINE_RE = new RegExp(String.raw`^(?<name>.*?)\s*(?:(?<unit>${PRICE})\s*[x×]\s*(?<qty>\d+)\s+)?(?<price>${PRICE})` +
    String.raw`\s*(?:€|EUR|E|e)?(?:\s*${L}|\s*\d)?$`, 'u');
  const TAIL_RE = new RegExp(String.raw`(?:\s+${PRICE}(?:\s*${L})?)+$`, 'u');
  const DETAIL_RE = /^(?:[A-ZÄÖÜ]{1,5}\s+)?\d+\s*(?:st|stk|stück|ml|g|mg|x)(?![\p{L}\d])/iu;
  const QTY_RE = /(?<![\p{L}\d])(\d+)\s*x(?![\p{L}\d])/iu;
  const TOTAL_RE = /^(?:end)?summe|^z\p{L}{0,2}\s*(?:be)?zahlen/iu;  // OCR liest „ZU“ gern als „ZUJ“/„ZIJ“
  const DEPOSIT_RETURN = ['pfandrück', 'pfandruck', 'leergut'];
  const starts = (s, list) => list.some(w => s.startsWith(w));
  const toCents = s => Math.round(parseFloat(s.replace(',', '.')) * 100);
  const item = (name, cents, qty = 1) => ({ name, cents, pfand: 0, discount: 0, qty });
  const cost = i => i.cents + i.pfand + (i.discount || 0);

  // Zeilen eines Bildes → {items, total}
  function parse(rows) {
    const items = [];
    let total = null, label = null;
    for (const raw of rows) {
      const text = raw.replace(/\s+/g, ' ').trim();
      const m = LINE_RE.exec(text);
      if (!m) {
        if (/\p{L}{3}/u.test(text)) label = text;
        continue;
      }
      let name = m.groups.name.replace(TAIL_RE, '').trim();
      const cents = toCents(m.groups.price);
      let qty = m.groups.qty ? +m.groups.qty : 1;
      if (label && DETAIL_RE.test(name) && !TOTAL_RE.test(name)) {
        const q = QTY_RE.exec(name);
        name = label;
        if (q) qty = +q[1];
      }
      label = null;
      const low = name.toLowerCase();
      if (TOTAL_RE.test(name)) { total = cents; break; }
      if (!name || !/\p{L}/u.test(name)) continue;
      if (starts(low, DEPOSIT_RETURN)) items.push(item(name, cents));
      else if (low.startsWith('pfand') && items.length) items[items.length - 1].pfand += cents;
      else if (cents < 0 && items.length) items[items.length - 1].discount += cents;  // Preisvorteil, Rabatt …
      else items.push(item(name, cents, qty));
    }
    return { items, total };
  }

  // Screenshots eines Bons zusammenfügen, Überlappungen entfernen
  function mergeScreens(parts) {
    const key = i => [i.name, i.cents, i.pfand, i.discount].join('|');
    let items = [], total = null;
    for (const p of parts) {
      let best = 0;
      for (let k = Math.min(items.length, p.items.length); k > 0; k--) {
        if (items.slice(-k).map(key).join('§') === p.items.slice(0, k).map(key).join('§')) { best = k; break; }
      }
      items = items.concat(p.items.slice(best));
      if (p.total !== null) total = p.total;
    }
    return { items, total };
  }

  // Ein Bon endet bei dem Bild, das eine Summe enthält
  function splitReceipts(parsed) {
    const out = [];
    let cur = [];
    for (const p of parsed) {
      cur.push(p);
      if (p.total !== null) { out.push(mergeScreens(cur)); cur = []; }
    }
    if (cur.length) out.push(mergeScreens(cur));
    return out;
  }

  return { parse, splitReceipts, cost };
})();
if (typeof module !== 'undefined') module.exports = Parse;
