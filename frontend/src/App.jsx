import { useState, useCallback, useRef, useEffect } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ─── 客户端 3D LUT 引擎（实时预览）──────────────────────────────────────
function parseCube(text) {
  let size = 0;
  const data = [];
  for (const ln of text.split("\n")) {
    const t = ln.trim();
    if (t.startsWith("LUT_3D_SIZE")) size = parseInt(t.split(/\s+/)[1]);
    else if (/^[0-9.]/.test(t)) {
      const p = t.split(/\s+/).map(Number);
      if (p.length === 3) data.push(p);
    }
  }
  return { size, data };
}

// 对 ImageData 应用 LUT（trilinear），返回新的 Float32 结果（[0,1] RGB）
function applyLutToImage(imgData, lut) {
  const { size: N, data } = lut;
  const px = imgData.data;
  const n = imgData.width * imgData.height;
  const out = new Float32Array(n * 3);
  const idx = (r, g, b) => r + g * N + b * N * N;
  for (let i = 0; i < n; i++) {
    const R = px[i * 4] / 255, G = px[i * 4 + 1] / 255, B = px[i * 4 + 2] / 255;
    const rf = R * (N - 1), gf = G * (N - 1), bf = B * (N - 1);
    const r0 = Math.floor(rf), g0 = Math.floor(gf), b0 = Math.floor(bf);
    const r1 = Math.min(r0 + 1, N - 1), g1 = Math.min(g0 + 1, N - 1), b1 = Math.min(b0 + 1, N - 1);
    const dr = rf - r0, dg = gf - g0, db = bf - b0;
    for (let ax = 0; ax < 3; ax++) {
      const c000 = data[idx(r0, g0, b0)][ax], c100 = data[idx(r1, g0, b0)][ax];
      const c010 = data[idx(r0, g1, b0)][ax], c110 = data[idx(r1, g1, b0)][ax];
      const c001 = data[idx(r0, g0, b1)][ax], c101 = data[idx(r1, g0, b1)][ax];
      const c011 = data[idx(r0, g1, b1)][ax], c111 = data[idx(r1, g1, b1)][ax];
      const c00 = c000 * (1 - dr) + c100 * dr, c10 = c010 * (1 - dr) + c110 * dr;
      const c01 = c001 * (1 - dr) + c101 * dr, c11 = c011 * (1 - dr) + c111 * dr;
      const c0 = c00 * (1 - dg) + c10 * dg, c1 = c01 * (1 - dg) + c11 * dg;
      out[i * 3 + ax] = c0 * (1 - db) + c1 * db;
    }
  }
  return out;
}

// ─── 强度混合 + 保护（高光/饱和度/阴影偏色）─────────────────────────────────
// o=原色, l=LUT输出色（均 [0,1] 三元组）, s=应用强度。
// s≤1 为纯插值（保持既定观感）；s>1 时对"外推过冲"做保护，防过曝/饱和过高/阴影偏色/断层。
const _LUMA = [0.2126, 0.7152, 0.0722];
const _smooth = (a, b, x) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
// 高光软拐点：>knee 平滑压向 1（渐近不超 1），消除硬裁剪导致的过曝与断层
const _softHi = v => { const k = 0.90; return v > k ? k + (1 - k) * Math.tanh((v - k) / (1 - k)) : v; };

function blendStrength(o, l, s) {
  let t0 = o[0] + s * (l[0] - o[0]), t1 = o[1] + s * (l[1] - o[1]), t2 = o[2] + s * (l[2] - o[2]);
  if (s > 1) {
    // 以 LUT 输出色（s=1 的既定观感）为参考，只驯服超出部分
    const Ll = _LUMA[0] * l[0] + _LUMA[1] * l[1] + _LUMA[2] * l[2];
    const Lt = _LUMA[0] * t0 + _LUMA[1] * t1 + _LUMA[2] * t2;
    let cl0 = l[0] - Ll, cl1 = l[1] - Ll, cl2 = l[2] - Ll;   // 参考彩度
    let ct0 = t0 - Lt, ct1 = t1 - Lt, ct2 = t2 - Lt;         // 当前彩度
    // 阴影偏色保护：暗部把彩度拉回既定值，抑制被放大的色偏
    const shw = _smooth(0.05, 0.25, Ll);
    ct0 = cl0 + shw * (ct0 - cl0); ct1 = cl1 + shw * (ct1 - cl1); ct2 = cl2 + shw * (ct2 - cl2);
    // 饱和度保护：限制彩度幅度相对既定值的增幅
    const magL = Math.sqrt(cl0 * cl0 + cl1 * cl1 + cl2 * cl2) + 1e-6;
    const magT = Math.sqrt(ct0 * ct0 + ct1 * ct1 + ct2 * ct2);
    const cap = magL * (1 + 0.5 * (s - 1)) + 0.05;
    if (magT > cap) { const k = cap / magT; ct0 *= k; ct1 *= k; ct2 *= k; }
    // 重组 + 高光软保护
    t0 = _softHi(Lt + ct0); t1 = _softHi(Lt + ct1); t2 = _softHi(Lt + ct2);
  }
  return [Math.max(0, Math.min(1, t0)), Math.max(0, Math.min(1, t1)), Math.max(0, Math.min(1, t2))];
}

// 深阴影守卫：原图本就深黑的像素，限制被抬亮 + 去掉强加的饱和色（防"阴影拉高染色/断层"）
// orig=原色, out=处理后色。仅在原图深阴影(Lo<~0.16)介入，中高调不受影响。
function shadowGuard(orig, out) {
  const Lo = 0.2126 * orig[0] + 0.7152 * orig[1] + 0.0722 * orig[2];
  const g = _smooth(0.02, 0.16, Lo);          // 0=原图纯黑 → 1=出阴影区
  if (g >= 0.999) return out;
  const Lout = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2];
  const maxL = Lo + 0.04 + 0.10 * g;          // 限制抬亮：纯黑最多到 0.04
  const Lc = Math.min(Lout, maxL);
  const desat = 0.35 + 0.65 * g;              // 去饱和：纯黑只留 35% 彩度
  return [
    Math.max(0, Math.min(1, Lc + (out[0] - Lout) * desat)),
    Math.max(0, Math.min(1, Lc + (out[1] - Lout) * desat)),
    Math.max(0, Math.min(1, Lc + (out[2] - Lout) * desat)),
  ];
}

// 三轴混合：颜色迁移(原图↔仅颜色LUT) + 影调迁移(仅颜色↔完整LUT) + 应用强度(整体) + 深阴影守卫
function stylePixel(orig, c0, c1, c2, f0, f1, f2, colorAmt, toneAmt, strength) {
  const s0 = orig[0] + colorAmt * (c0 - orig[0]) + toneAmt * (f0 - c0);
  const s1 = orig[1] + colorAmt * (c1 - orig[1]) + toneAmt * (f1 - c1);
  const s2 = orig[2] + colorAmt * (c2 - orig[2]) + toneAmt * (f2 - c2);
  const out = [
    Math.max(0, Math.min(1, orig[0] + strength * (s0 - orig[0]))),
    Math.max(0, Math.min(1, orig[1] + strength * (s1 - orig[1]))),
    Math.max(0, Math.min(1, orig[2] + strength * (s2 - orig[2]))),
  ];
  return shadowGuard(orig, out);
}

// ─── 还原补偿：亮度全局仿射 + 按明暗分档的色度偏移（还原色调分离，不发灰）──
// 亮度对齐参考整体明暗/对比；色度按亮度档分别拉向参考同档颜色（阴影/高光各自的色）
const _LUMA_JS = [0.2126, 0.7152, 0.0722];
// 单调曲线插值（xs 分位数升序 → ys 目标），用于亮度曲线匹配
function interpL(x, xs, ys) {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo] || 1);
  return ys[lo] + t * (ys[hi] - ys[lo]);
}
function _interpDelta(L, bins, delta) {
  const n = bins.length;
  if (L <= bins[0]) return delta[0];
  if (L >= bins[n - 1]) return delta[n - 1];
  let i = 0; while (i < n - 1 && bins[i + 1] < L) i++;
  const t = (L - bins[i]) / (bins[i + 1] - bins[i]);
  return [0, 1, 2].map(k => delta[i][k] + t * (delta[i + 1][k] - delta[i][k]));
}
function reproCorrect(c, st, w) {
  if (!st || w <= 0) return c;
  const L = _LUMA_JS[0] * c[0] + _LUMA_JS[1] * c[1] + _LUMA_JS[2] * c[2];
  // 曝光对齐原图：用【乘性增益】而非加性平移——0×gain=0 天然保黑，绝不抬黑
  // （情况B：参考暗是其内容暗，只把整体曝光拉回原图，不照搬参考绝对明暗）
  const mid = st.cLq.length >> 1;
  const cMed = st.cLq[mid], target = st.srcLmed;
  const gain = Math.max(0.5, Math.min(target / Math.max(cMed, 0.06), 2.2));
  let Lc = L * (1 + w * (gain - 1));
  // 压高光 / 压暗阴影（黑处为负 → 更黑，不抬黑）；软高光防过曝
  const extra = -0.10 * _smooth(0.55, 1.0, L) - 0.07 * (1 - _smooth(0.0, 0.35, L));
  Lc = _softHi(Math.max(0, Lc + w * extra));
  // 色度偏移 + 幅度上限（情况B 防参考内容色整体染色）
  let d0 = 0, d1 = 0, d2 = 0;
  { const d = _interpDelta(L, st.bins, st.delta); d0 = d[0]; d1 = d[1]; d2 = d[2]; }
  const dmag = Math.sqrt(d0 * d0 + d1 * d1 + d2 * d2);
  if (dmag > 0.12) { const k = 0.12 / dmag; d0 *= k; d1 *= k; d2 *= k; }
  // 极值衰减：高光/近黑处减弱色度校正（避免高光染色偏黄、暗部糊死）
  const attL = (1 - 0.85 * _smooth(0.72, 0.98, L)) * _smooth(0.0, 0.05, L);
  // 饱和度衰减：已鲜艳的颜色少校正，保留其色相（防绿变黄等）
  const cr = c[0] - L, cg = c[1] - L, cb = c[2] - L;
  const mag = Math.sqrt(cr * cr + cg * cg + cb * cb);
  const attS = 1 - 0.75 * _smooth(0.10, 0.35, mag);
  const att = attL * attS * w;
  return [
    Math.max(0, Math.min(1, Lc + cr + att * d0)),
    Math.max(0, Math.min(1, Lc + cg + att * d1)),
    Math.max(0, Math.min(1, Lc + cb + att * d2)),
  ];
}
// 从后端 repro 统计量预计算校正参数
function reproParams(repro) {
  if (!repro) return null;
  return { cLq: repro.cnn_Lq, rLq: repro.ref_Lq, srcLmed: repro.src_Lmed, bins: repro.bins, delta: repro.delta };
}

const COLORS = {
  bg:          "#0a0a0a",
  surface:     "#111111",
  border:      "#1e1e1e",
  borderHover: "#2e2e2e",
  accent:      "#c8a96e",
  accentDim:   "#8a7048",
  text:        "#e8e8e8",
  textMuted:   "#666",
  textDim:     "#999",
  success:     "#4a9e6e",
  warning:     "#c8843a",
  error:       "#9e4a4a",
};

export default function App() {
  const [refFile,     setRefFile]     = useState(null);
  const [srcFile,     setSrcFile]     = useState(null);
  const [refPreview,  setRefPreview]  = useState(null);
  const [srcPreview,  setSrcPreview]  = useState(null);
  const [presetName,  setPresetName]  = useState("AI Style");
  const [strength,    setStrength]    = useState(1.0);  // 迁移强度：0=原图，1=完整迁移，>1=加强
  const [modeChoice,  setModeChoice]  = useState("auto"); // auto / A(精确复刻同图) / B(色相迁移)
  const [loading,     setLoading]     = useState(false);
  const [result,      setResult]      = useState(null);
  const [error,       setError]       = useState(null);
  const [activeTab,   setActiveTab]   = useState("report");

  const refInputRef = useRef();
  const srcInputRef = useRef();
  const previewCanvas = useRef();
  // 预计算缓冲：原图像素 + LUT 全量结果（strength 滑块只做混合，实时）
  const bufs = useRef(null);   // { w, h, orig: Float32, lut: Float32 }

  // 结果就绪时：加载原图 → 预计算 orig + LUT 全量结果（strength 滑块只做混合，实时）
  useEffect(() => {
    if (!result?.lut_content || !srcPreview) { bufs.current = null; return; }
    const img = new Image();
    img.onload = () => {
      const maxW = 1000;  // 预览分辨率（仅屏显；下载的 LUT 分辨率无关，套用时全分辨率）
      const scale = Math.min(1, maxW / img.width);
      const w = Math.round(img.width * scale), h = Math.round(img.height * scale);
      const cv = document.createElement("canvas");
      cv.width = w; cv.height = h;
      const ctx = cv.getContext("2d");
      ctx.drawImage(img, 0, 0, w, h);
      const imgData = ctx.getImageData(0, 0, w, h);
      const orig = new Float32Array(w * h * 3);
      for (let i = 0; i < w * h; i++) {
        orig[i * 3] = imgData.data[i * 4] / 255;
        orig[i * 3 + 1] = imgData.data[i * 4 + 1] / 255;
        orig[i * 3 + 2] = imgData.data[i * 4 + 2] / 255;
      }
      const lut = applyLutToImage(imgData, parseCube(result.lut_content));
      bufs.current = { w, h, orig, lut };
      renderPreview(strength);
    };
    img.src = srcPreview;
  }, [result, srcPreview]);

  // 迁移强度变化时实时渲染
  useEffect(() => { renderPreview(strength); }, [strength]);

  const renderPreview = (s) => {
    const b = bufs.current, cv = previewCanvas.current;
    if (!b || !cv) return;
    cv.width = b.w; cv.height = b.h;
    const ctx = cv.getContext("2d");
    const out = ctx.createImageData(b.w, b.h);
    for (let i = 0; i < b.w * b.h; i++) {
      for (let ax = 0; ax < 3; ax++) {
        const v = b.orig[i * 3 + ax] * (1 - s) + b.lut[i * 3 + ax] * s;  // 原图↔迁移 混合
        out.data[i * 4 + ax] = Math.max(0, Math.min(255, v * 255));
      }
      out.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
  };

  const handleFile = useCallback((file, type) => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    if (type === "ref") { setRefFile(file); setRefPreview(url); }
    else                { setSrcFile(file); setSrcPreview(url); }
    setResult(null);
    setError(null);
  }, []);

  const onDrop = useCallback((e, type) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file, type);
  }, [handleFile]);

  // 打开页面即预热后端（Render 免费层休眠后需 ~40s 唤醒）
  useEffect(() => { fetch(`${API_BASE}/health`).catch(() => {}); }, []);

  const analyze = async () => {
    if (!refFile || !srcFile) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const form = new FormData();
    form.append("src_image",    srcFile);
    form.append("ref_image",    refFile);
    form.append("preset_name",  presetName);
    form.append("mode",         modeChoice);

    // 冷启动可能失败/超时，自动重试（每次间隔递增，覆盖 ~40s 唤醒窗口）
    const submit = async () => {
      const res = await fetch(`${API_BASE}/analyze`, { method: "POST", body: form });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
      return res.json();
    };
    const delays = [0, 5000, 8000, 12000];
    let lastErr;
    for (let i = 0; i < delays.length; i++) {
      if (delays[i]) {
        setError(`后端启动中，请稍候…（${i}/3 重试）`);
        await new Promise(r => setTimeout(r, delays[i]));
      }
      try {
        const data = await submit();
        setResult(data);
        setActiveTab("report");
        setError(null);
        setLoading(false);
        return;
      } catch (e) { lastErr = e; }
    }
    setError(`生成失败：${lastErr?.message || "网络错误"}（若后端刚唤醒，请再点一次）`);
    setLoading(false);
  };

  const download = (content, ext, mime) => {
    if (!content) return;
    const blob = new Blob([content], { type: mime });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `${presetName}.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
  };
  // 把迁移强度烘焙进 .cube（LUT 向 identity 混合，与预览一致）
  const lutWithStrength = () => {
    if (!result?.lut_content) return null;
    if (strength === 1.0) return result.lut_content;
    const { size: N, data } = parseCube(result.lut_content);
    const lines = [];
    for (const l of result.lut_content.split("\n")) {
      if (/^[0-9.]/.test(l.trim())) break;
      lines.push(l);
    }
    for (let i = 0; i < data.length; i++) {
      const r = i % N, g = Math.floor(i / N) % N, b = Math.floor(i / (N * N));
      const id = [r / (N - 1), g / (N - 1), b / (N - 1)];
      const v = data[i].map((x, k) => Math.max(0, Math.min(1, id[k] * (1 - strength) + x * strength)));
    }
    return lines.join("\n") + "\n";
  };
  const downloadLut = () => download(lutWithStrength(), "cube", "text/plain");

  return (
    <div style={{ minHeight: "100vh", background: COLORS.bg, color: COLORS.text, fontFamily: "'Georgia','Times New Roman',serif" }}>

      {/* Header */}
      <header style={{ borderBottom: `1px solid ${COLORS.border}`, padding: "28px 48px", display: "flex", alignItems: "baseline", gap: "16px" }}>
        <h1 style={{ margin: 0, fontSize: "22px", fontWeight: "400", letterSpacing: "0.12em", color: COLORS.accent, fontFamily: "'Georgia',serif" }}>
          风格移植 · LUT 生成器
        </h1>
        <span style={{ color: COLORS.textMuted, fontSize: "12px", letterSpacing: "0.08em", fontFamily: "monospace" }}>
          原图 + 风格参考 → 通用 3D LUT (.cube)
        </span>
      </header>

      <main style={{ maxWidth: "1100px", margin: "0 auto", padding: "48px" }}>

        {/* Image Upload */}
        <section style={{ marginBottom: "28px" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", marginBottom: "16px" }}>
            <DropZone label="参考图" sublabel="必须 · 风格来源"
              preview={refPreview} accent
              onDrop={e => onDrop(e, "ref")}
              onClick={() => refInputRef.current?.click()} />
            <input ref={refInputRef} type="file"
              accept="image/*,.cr2,.cr3,.nef,.arw,.raf,.dng,.rw2"
              style={{ display: "none" }}
              onChange={e => handleFile(e.target.files[0], "ref")} />

            <DropZone label="原图" sublabel="必须 · 你要调色的照片" accent
              preview={srcPreview}
              onDrop={e => onDrop(e, "src")}
              onClick={() => srcInputRef.current?.click()} />
            <input ref={srcInputRef} type="file"
              accept="image/*,.cr2,.cr3,.nef,.arw,.raf,.dng,.rw2"
              style={{ display: "none" }}
              onChange={e => handleFile(e.target.files[0], "src")} />
          </div>

          {/* Mode selector */}
          <div style={{ marginBottom: "12px" }}>
            <FieldLabel>迁移模式（自动判别不准时手动指定）</FieldLabel>
            <div style={{ display: "flex", gap: "6px", marginTop: "5px" }}>
              {[["auto", "自动"], ["A", "精确复刻（同一张图）"], ["B", "色相迁移（不同照片）"]].map(([v, label]) => (
                <button key={v} onClick={() => setModeChoice(v)} style={{
                  flex: 1, padding: "7px 8px", fontSize: "11px", fontFamily: "monospace",
                  cursor: "pointer", letterSpacing: "0.04em",
                  background: modeChoice === v ? COLORS.accentDim : COLORS.surface,
                  color: modeChoice === v ? "#000" : COLORS.textDim,
                  border: `1px solid ${modeChoice === v ? COLORS.accent : COLORS.border}`,
                }}>{label}</button>
              ))}
            </div>
          </div>

          {/* Options row */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "16px", alignItems: "center", marginBottom: "10px" }}>
            <div>
              <FieldLabel>预设名称</FieldLabel>
              <input
                value={presetName}
                onChange={e => setPresetName(e.target.value)}
                style={{
                  width: "100%", boxSizing: "border-box",
                  background: COLORS.surface, border: `1px solid ${COLORS.border}`,
                  color: COLORS.text, padding: "8px 12px",
                  fontSize: "12px", fontFamily: "monospace", outline: "none",
                }}
              />
            </div>

            <div style={{ paddingTop: "18px" }}>
              <button
                onClick={analyze}
                disabled={!refFile || !srcFile || loading}
                style={{
                  background: refFile && srcFile && !loading ? COLORS.accent : COLORS.border,
                  color:      refFile && srcFile && !loading ? "#000" : COLORS.textMuted,
                  border: "none", padding: "8px 28px",
                  fontSize: "12px", letterSpacing: "0.12em",
                  fontFamily: "'Georgia',serif",
                  cursor: refFile && srcFile && !loading ? "pointer" : "not-allowed",
                  whiteSpace: "nowrap",
                }}
              >
                {loading ? "生成中..." : "生成 LUT"}
              </button>
            </div>
          </div>

          {/* Mode indicator */}
          {(refFile || srcFile) && (
            <div style={{ fontSize: "11px", color: COLORS.textMuted, fontFamily: "monospace", letterSpacing: "0.06em" }}>
              {srcFile && refFile ? "● 就绪 — 点击生成，实时预览 + 下载 LUT" : "● 请上传原图 + 风格参考图"}
            </div>
          )}
        </section>

        {/* Error */}
        {error && (
          <div style={{ background: "#1a0a0a", border: `1px solid ${COLORS.error}`, padding: "14px 20px", marginBottom: "24px", fontSize: "13px", color: "#e08080", fontFamily: "monospace" }}>
            ✗ {error}
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div style={{ textAlign: "center", padding: "60px 0" }}>
            <LoadingDots />
            <div style={{ marginTop: "16px", color: COLORS.textMuted, fontSize: "12px", letterSpacing: "0.1em", fontFamily: "monospace" }}>
              正在分析色彩风格...
            </div>
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <section>
            {/* 实时效果预览（双图 LUT 模式）*/}
            {result.lut_content && srcPreview && (
              <div style={{ marginBottom: "28px", background: COLORS.surface, border: `1px solid ${COLORS.border}`, padding: "20px" }}>
                {result.mode && (
                  <div style={{ marginBottom: "12px", fontSize: "11px", fontFamily: "monospace", letterSpacing: "0.06em", color: COLORS.accent }}>
                    {result.mode === "A"
                      ? "● 情况A · 空间对应 — 同一张图，精确复刻参考调色（含色相旋转）"
                      : "● 情况B · 按色相匹配 — 色相 band 内小幅微调(不跨band塌陷) + 饱和/亮度向参考倾斜(保对比)"}
                  </div>
                )}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1.4fr", gap: "12px", alignItems: "start" }}>
                  <div>
                    <FieldLabel>原图</FieldLabel>
                    <img src={srcPreview} style={{ width: "100%", marginTop: "6px", border: `1px solid ${COLORS.border}` }} />
                  </div>
                  <div>
                    <FieldLabel>参考风格</FieldLabel>
                    <img src={refPreview} style={{ width: "100%", marginTop: "6px", border: `1px solid ${COLORS.border}` }} />
                  </div>
                  <div>
                    <FieldLabel>应用效果（实时）· 强度 {Math.round(strength * 100)}%</FieldLabel>
                    <canvas ref={previewCanvas} style={{ width: "100%", marginTop: "6px", border: `1px solid ${COLORS.accent}`, display: "block" }} />
                  </div>
                </div>
                {/* 迁移强度 */}
                <div style={{ marginTop: "16px" }}>
                  <Slider label="迁移强度" value={strength} min={0} max={1.5} step={0.05}
                    display={`${Math.round(strength * 100)}%`}
                    onChange={setStrength}
                    ends={["0% 原图", "100% 标准 · 150% 加强"]} />
                </div>
              </div>
            )}

            {/* Tabs */}
            <div style={{ display: "flex", borderBottom: `1px solid ${COLORS.border}`, marginBottom: "28px" }}>
              {["report", "params"].map(tab => (
                <button key={tab} onClick={() => setActiveTab(tab)} style={{
                  background: "none", border: "none",
                  borderBottom: activeTab === tab ? `1px solid ${COLORS.accent}` : "1px solid transparent",
                  color:        activeTab === tab ? COLORS.accent : COLORS.textMuted,
                  padding: "10px 24px", fontSize: "11px", letterSpacing: "0.12em",
                  fontFamily: "monospace", cursor: "pointer", marginBottom: "-1px",
                }}>
                  {{ report: "分析报告", params: "参数详情" }[tab]}
                </button>
              ))}
              <div style={{ marginLeft: "auto", paddingBottom: "4px" }}>
                <button onClick={downloadLut} style={{
                  background: COLORS.accent, border: `1px solid ${COLORS.accent}`,
                  color: "#000", padding: "6px 24px",
                  fontSize: "11px", letterSpacing: "0.1em",
                  fontFamily: "monospace", cursor: "pointer",
                }}>
                  ↓ 下载 .cube LUT
                </button>
              </div>
            </div>

            {activeTab === "report" && <ReportTab result={result} />}
            {activeTab === "params" && <ParamsTab summary={result.summary} />}
          </section>
        )}
      </main>
    </div>
  );
}

// ── Sub-components ──────────────────────────────────────────────

function DropZone({ label, sublabel, preview, onDrop, onClick, accent }) {
  const [hovering, setHovering] = useState(false);
  return (
    <div
      onClick={onClick}
      onDrop={onDrop}
      onDragOver={e => { e.preventDefault(); setHovering(true); }}
      onDragLeave={() => setHovering(false)}
      style={{
        border: `1px solid ${hovering ? (accent ? COLORS.accent : COLORS.borderHover) : COLORS.border}`,
        background: COLORS.surface, cursor: "pointer",
        height: "200px", display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        position: "relative", overflow: "hidden", transition: "border-color 0.2s",
      }}
    >
      {preview ? (
        <>
          <img src={preview} alt="" style={{ width: "100%", height: "100%", objectFit: "cover", opacity: 0.9 }} />
          <div style={{
            position: "absolute", bottom: 0, left: 0, right: 0,
            background: "linear-gradient(transparent,rgba(0,0,0,0.8))",
            padding: "20px 12px 8px", fontSize: "10px",
            letterSpacing: "0.1em", fontFamily: "monospace", color: "#ccc",
          }}>{label}</div>
        </>
      ) : (
        <>
          <div style={{ fontSize: "28px", color: accent ? COLORS.accentDim : COLORS.border, marginBottom: "8px" }}>+</div>
          <div style={{ fontSize: "12px", letterSpacing: "0.1em", color: accent ? COLORS.accent : COLORS.textMuted, fontFamily: "monospace" }}>{label}</div>
          <div style={{ fontSize: "10px", color: COLORS.textMuted, marginTop: "4px", fontFamily: "monospace" }}>{sublabel}</div>
        </>
      )}
    </div>
  );
}

function Slider({ label, value, min, max, step, display, onChange, ends }) {
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", fontFamily: "monospace", color: COLORS.textDim, marginBottom: "4px" }}>
        <span>{label}</span>
        <span style={{ color: COLORS.accent }}>{display}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: "100%", accentColor: COLORS.accent, cursor: "pointer" }} />
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "9px", fontFamily: "monospace", color: COLORS.textMuted, marginTop: "2px" }}>
        <span>{ends[0]}</span><span>{ends[1]}</span>
      </div>
    </div>
  );
}

function FieldLabel({ children }) {
  return (
    <div style={{ fontSize: "9px", letterSpacing: "0.16em", color: COLORS.textMuted, fontFamily: "monospace", marginBottom: "5px" }}>
      {children}
    </div>
  );
}

function ReportTab({ result }) {
  const groups = Object.entries(result.summary || {}).filter(([, v]) => Object.keys(v).length);
  return (
    <div style={{ background: COLORS.surface, border: `1px solid ${COLORS.border}`, padding: "24px" }}>
      <Label>风格摘要 · 按色相匹配的颜色迁移</Label>
      <div style={{ marginTop: "10px" }}>
        {groups.length === 0
          ? <div style={{ fontSize: "12px", color: COLORS.textMuted, fontFamily: "monospace" }}>接近原图（变化很小）</div>
          : groups.map(([name, params]) => (
              <Row key={name} label={name} value={`${Object.keys(params).length} 项调整`} />
            ))}
      </div>
    </div>
  );
}

function ParamsTab({ summary }) {
  if (!summary) return null;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "16px" }}>
      {Object.entries(summary).map(([section, params]) => (
        <div key={section} style={{ background: COLORS.surface, border: `1px solid ${COLORS.border}`, padding: "20px" }}>
          <Label>{section}</Label>
          <div style={{ marginTop: "10px" }}>
            {Object.entries(params).map(([k, v]) => (
              <div key={k} style={{
                display: "flex", justifyContent: "space-between", alignItems: "center",
                padding: "5px 0", borderBottom: `1px solid ${COLORS.border}`,
                fontSize: "12px", fontFamily: "monospace",
              }}>
                <span style={{ color: COLORS.textMuted }}>{k}</span>
                <span style={{ color: typeof v === "number" && v !== 0 ? (v > 0 ? "#8ec89e" : "#c88e8e") : COLORS.textDim, fontWeight: "500" }}>
                  {typeof v === "number" ? (v > 0 ? `+${v}` : `${v}`) : v}
                </span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function Label({ children }) {
  return (
    <div style={{ fontSize: "9px", letterSpacing: "0.18em", color: COLORS.textMuted, fontFamily: "monospace", textTransform: "uppercase" }}>
      {children}
    </div>
  );
}

function Row({ label, value, warn, good }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: `1px solid ${COLORS.border}`, fontSize: "12px", fontFamily: "monospace" }}>
      <span style={{ color: COLORS.textMuted }}>{label}</span>
      <span style={{ color: warn ? COLORS.warning : good ? COLORS.success : COLORS.textDim }}>{value}</span>
    </div>
  );
}

function LoadingDots() {
  return (
    <div style={{ display: "flex", gap: "6px", justifyContent: "center" }}>
      {[0, 1, 2].map(i => (
        <div key={i} style={{
          width: "6px", height: "6px", background: COLORS.accent,
          borderRadius: "50%", animation: `pulse 1.2s ease-in-out ${i * 0.2}s infinite`,
        }} />
      ))}
      <style>{`@keyframes pulse { 0%,80%,100%{opacity:.2;transform:scale(.8)} 40%{opacity:1;transform:scale(1)} }`}</style>
    </div>
  );
}
