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

// ─── LAB 色彩空间 + 解析匹配（与后端 numpy 实现一致，D65）──────────────────
const _EPS = 216 / 24389, _KAPPA = 24389 / 27;
const _srgb2lin = c => c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
const _lin2srgb = c => { c = Math.max(0, Math.min(1, c)); return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055; };
function rgb2lab(r, g, b) {
  const rl = _srgb2lin(r), gl = _srgb2lin(g), bl = _srgb2lin(b);
  const X = (rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375) / 0.95047;
  const Y = (rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750);
  const Z = (rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041) / 1.08883;
  const fx = X > _EPS ? Math.cbrt(X) : (_KAPPA * X + 16) / 116;
  const fy = Y > _EPS ? Math.cbrt(Y) : (_KAPPA * Y + 16) / 116;
  const fz = Z > _EPS ? Math.cbrt(Z) : (_KAPPA * Z + 16) / 116;
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}
function lab2rgb(L, a, b) {
  const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
  const fx3 = fx * fx * fx, fy3 = fy * fy * fy, fz3 = fz * fz * fz;
  const xr = (fx3 > _EPS ? fx3 : (116 * fx - 16) / _KAPPA) * 0.95047;
  const yr = (L > _KAPPA * _EPS ? fy3 : L / _KAPPA);
  const zr = (fz3 > _EPS ? fz3 : (116 * fz - 16) / _KAPPA) * 1.08883;
  const rl = xr * 3.2404542 + yr * -1.5371385 + zr * -0.4985314;
  const gl = xr * -0.9692660 + yr * 1.8760108 + zr * 0.0415560;
  const bl = xr * 0.0556434 + yr * -0.2040259 + zr * 1.0572252;
  return [_lin2srgb(rl), _lin2srgb(gl), _lin2srgb(bl)];
}
// L 分位数插值（xs, ys 均单调升）
function interpL(x, xs, ys) {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo] || 1);
  return ys[lo] + t * (ys[hi] - ys[lo]);
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
  const [strength,    setStrength]    = useState(1.0);  // 风格应用强度（LUT 不透明度）
  const [matchWeight, setMatchWeight] = useState(0.7);  // 忠实还原(1) ↔ LR风格(0)
  const [colorStr,    setColorStr]    = useState(0.85); // 色彩迁移强度（抑制溢色）
  const [loading,     setLoading]     = useState(false);
  const [result,      setResult]      = useState(null);
  const [error,       setError]       = useState(null);
  const [activeTab,   setActiveTab]   = useState("report");

  const refInputRef = useRef();
  const srcInputRef = useRef();
  const previewCanvas = useRef();
  // 预计算缓冲：原图像素 + LUT 全量结果（strength 滑块只做混合，实时）
  const bufs = useRef(null);   // { w, h, orig: Float32, lut: Float32 }

  // 结果就绪时：加载原图 → 预计算 orig / CNN结果 / 解析匹配的 LAB 基（L',a0,b0）
  useEffect(() => {
    if (!result?.lut_style || !result?.match_stats || !srcPreview) { bufs.current = null; return; }
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
      const n = w * h;
      const st = result.match_stats;
      const sLq = st.src_Lq, rLq = st.ref_Lq, sm = st.src_ab_mean, asc = st.ab_scale;
      const orig = new Float32Array(n * 3);
      const Lp = new Float32Array(n), a0 = new Float32Array(n), b0 = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const r = imgData.data[i * 4] / 255, g = imgData.data[i * 4 + 1] / 255, bl = imgData.data[i * 4 + 2] / 255;
        orig[i * 3] = r; orig[i * 3 + 1] = g; orig[i * 3 + 2] = bl;
        const [L, a, b] = rgb2lab(r, g, bl);
        Lp[i] = interpL(L, sLq, rLq);            // 影调匹配后的 L
        a0[i] = (a - sm[0]) * asc[0];            // a/b 方差匹配（色彩强度=0 的基）
        b0[i] = (b - sm[1]) * asc[1];
      }
      const cnn = applyLutToImage(imgData, parseCube(result.lut_style));  // CNN 分支
      bufs.current = { w, h, orig, cnn, Lp, a0, b0, st };
      renderPreview(strength, matchWeight, colorStr);
    };
    img.src = srcPreview;
  }, [result, srcPreview]);

  // 任一滑块变化 → 实时重渲染
  useEffect(() => { renderPreview(strength, matchWeight, colorStr); }, [strength, matchWeight, colorStr]);

  const renderPreview = (s, mw, cs) => {
    const b = bufs.current, cv = previewCanvas.current;
    if (!b || !cv) return;
    const { st } = b, sm = st.src_ab_mean, rm = st.ref_ab_mean;
    const da = sm[0] + cs * (rm[0] - sm[0]), db = sm[1] + cs * (rm[1] - sm[1]);
    cv.width = b.w; cv.height = b.h;
    const ctx = cv.getContext("2d");
    const out = ctx.createImageData(b.w, b.h);
    for (let i = 0; i < b.w * b.h; i++) {
      const A = lab2rgb(b.Lp[i], b.a0[i] + da, b.b0[i] + db);  // 解析匹配分支
      for (let ax = 0; ax < 3; ax++) {
        const blended = mw * A[ax] + (1 - mw) * b.cnn[i * 3 + ax];  // 忠实↔风格 混合
        const v = b.orig[i * 3 + ax] * (1 - s) + blended * s;       // 应用强度
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
  // 按当前三个滑块，在客户端烘焙最终 .cube（与预览完全一致）
  const buildCube = () => {
    if (!result?.lut_style || !result?.match_stats) return null;
    const { size: N, data } = parseCube(result.lut_style);   // CNN 分支（identity 网格）
    const st = result.match_stats;
    const sLq = st.src_Lq, rLq = st.ref_Lq, sm = st.src_ab_mean, rm = st.ref_ab_mean, asc = st.ab_scale;
    const da = sm[0] + colorStr * (rm[0] - sm[0]), db = sm[1] + colorStr * (rm[1] - sm[1]);
    const lines = [
      `TITLE "${presetName}"`, `LUT_3D_SIZE ${N}`,
      "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0", "",
    ];
    for (let i = 0; i < data.length; i++) {
      const r = (i % N) / (N - 1), g = (Math.floor(i / N) % N) / (N - 1), bl = Math.floor(i / (N * N)) / (N - 1);
      const [L, a, b] = rgb2lab(r, g, bl);
      const A = lab2rgb(interpL(L, sLq, rLq), (a - sm[0]) * asc[0] + da, (b - sm[1]) * asc[1] + db);
      const id = [r, g, bl];
      const v = [0, 1, 2].map(k => {
        const blended = matchWeight * A[k] + (1 - matchWeight) * data[i][k];
        return Math.max(0, Math.min(1, id[k] * (1 - strength) + blended * strength));
      });
      lines.push(`${v[0].toFixed(6)} ${v[1].toFixed(6)} ${v[2].toFixed(6)}`);
    }
    return lines.join("\n") + "\n";
  };
  const downloadLut = () => download(buildCube(), "cube", "text/plain");

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
            {result.lut_style && srcPreview && (
              <div style={{ marginBottom: "28px", background: COLORS.surface, border: `1px solid ${COLORS.border}`, padding: "20px" }}>
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
                {/* 三个滑块：忠实↔风格、色彩迁移、应用强度 */}
                <div style={{ marginTop: "16px", display: "grid", gap: "14px" }}>
                  <Slider label="忠实还原 ↔ LR 风格" value={matchWeight} min={0} max={1} step={0.05}
                    display={`还原 ${Math.round(matchWeight * 100)}%`}
                    onChange={setMatchWeight}
                    ends={["← LR 风格（分色/不溢色）", "忠实还原（整体照搬）→"]} />
                  <Slider label="色彩迁移强度" value={colorStr} min={0} max={1} step={0.05}
                    display={`${Math.round(colorStr * 100)}%`}
                    onChange={setColorStr}
                    ends={["0% 中性（抑制溢色）", "100% 完全对齐参考色"]} />
                  <Slider label="风格应用强度" value={strength} min={0} max={1.5} step={0.05}
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
      <Label>风格摘要 · CNN 预测的颜色变换</Label>
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
