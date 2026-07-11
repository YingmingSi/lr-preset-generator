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
  const [presetName,  setPresetName]  = useState("AI生成预设");
  const [strength,    setStrength]    = useState(1.0);  // 风格应用强度（LUT 不透明度）
  const [loading,     setLoading]     = useState(false);
  const [result,      setResult]      = useState(null);
  const [error,       setError]       = useState(null);
  const [activeTab,   setActiveTab]   = useState("report");

  const refInputRef = useRef();
  const srcInputRef = useRef();
  const previewCanvas = useRef();
  // 预计算缓冲：原图像素 + LUT 全量结果（strength 滑块只做混合，实时）
  const bufs = useRef(null);   // { w, h, orig: Float32, lut: Float32 }

  // 结果就绪时：加载原图 → 预计算 orig + LUT 全量结果
  useEffect(() => {
    if (!result?.lut_content || !srcPreview) { bufs.current = null; return; }
    const img = new Image();
    img.onload = () => {
      const maxW = 420;
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

  // strength 变化时实时混合渲染
  useEffect(() => { renderPreview(strength); }, [strength]);

  const renderPreview = (s) => {
    const b = bufs.current, cv = previewCanvas.current;
    if (!b || !cv) return;
    cv.width = b.w; cv.height = b.h;
    const ctx = cv.getContext("2d");
    const out = ctx.createImageData(b.w, b.h);
    for (let i = 0; i < b.w * b.h; i++) {
      for (let ax = 0; ax < 3; ax++) {
        const v = b.orig[i * 3 + ax] * (1 - s) + b.lut[i * 3 + ax] * s;
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

  const analyze = async () => {
    if (!refFile) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const form = new FormData();
    form.append("ref_image",    refFile);
    if (srcFile) form.append("src_image", srcFile);
    form.append("preset_name",  presetName);

    try {
      const res = await fetch(`${API_BASE}/analyze`, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "分析失败");
      }
      const data = await res.json();
      setResult(data);
      setActiveTab("report");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
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
  // 把风格强度烘焙进 .cube（LUT 向 identity 混合）
  const lutWithStrength = () => {
    if (!result?.lut_content) return null;
    if (strength === 1.0) return result.lut_content;
    const { size: N, data } = parseCube(result.lut_content);
    const header = result.lut_content.split("\n").filter(l => !/^[0-9.]/.test(l.trim()) || l.trim() === "");
    const lines = [];
    for (const l of result.lut_content.split("\n")) {
      if (/^[0-9.]/.test(l.trim())) break;
      lines.push(l);
    }
    for (let i = 0; i < data.length; i++) {
      const r = i % N, g = Math.floor(i / N) % N, b = Math.floor(i / (N * N));
      const id = [r / (N - 1), g / (N - 1), b / (N - 1)];
      const v = data[i].map((x, k) => Math.max(0, Math.min(1, id[k] * (1 - strength) + x * strength)));
      lines.push(`${v[0].toFixed(6)} ${v[1].toFixed(6)} ${v[2].toFixed(6)}`);
    }
    return lines.join("\n") + "\n";
  };
  const downloadLut = () => download(lutWithStrength(), "cube", "text/plain");
  const downloadXmp = () => download(result?.xmp_content, "xmp", "application/xml");

  return (
    <div style={{ minHeight: "100vh", background: COLORS.bg, color: COLORS.text, fontFamily: "'Georgia','Times New Roman',serif" }}>

      {/* Header */}
      <header style={{ borderBottom: `1px solid ${COLORS.border}`, padding: "28px 48px", display: "flex", alignItems: "baseline", gap: "16px" }}>
        <h1 style={{ margin: 0, fontSize: "22px", fontWeight: "400", letterSpacing: "0.12em", color: COLORS.accent, fontFamily: "'Georgia',serif" }}>
          LR · 预设生成器
        </h1>
        <span style={{ color: COLORS.textMuted, fontSize: "12px", letterSpacing: "0.08em", fontFamily: "monospace" }}>
          CNN 驱动 · 22 维参数预测 · R² 0.73
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

            <DropZone label="原图" sublabel="可选 · 支持RAW格式"
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
                disabled={!refFile || loading}
                style={{
                  background: refFile && !loading ? COLORS.accent : COLORS.border,
                  color:      refFile && !loading ? "#000" : COLORS.textMuted,
                  border: "none", padding: "8px 28px",
                  fontSize: "12px", letterSpacing: "0.12em",
                  fontFamily: "'Georgia',serif",
                  cursor: refFile && !loading ? "pointer" : "not-allowed",
                  whiteSpace: "nowrap",
                }}
              >
                {loading ? "分析中..." : "生成预设"}
              </button>
            </div>
          </div>

          {/* Mode indicator */}
          {(refFile || srcFile) && (
            <div style={{ fontSize: "11px", color: COLORS.textMuted, fontFamily: "monospace", letterSpacing: "0.06em" }}>
              {srcFile ? "● 双图模式 — CNN 风格移植 · 实时 LUT 预览" : "● 单图模式 — 风格特征提取"}
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
                {/* 风格强度滑块 */}
                <div style={{ marginTop: "16px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", fontFamily: "monospace", color: COLORS.textDim, marginBottom: "4px" }}>
                    <span>风格应用强度</span>
                    <span style={{ color: COLORS.accent }}>{Math.round(strength * 100)}%</span>
                  </div>
                  <input type="range" min="0" max="1.5" step="0.05" value={strength}
                    onChange={e => setStrength(parseFloat(e.target.value))}
                    style={{ width: "100%", accentColor: COLORS.accent, cursor: "pointer" }} />
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "9px", fontFamily: "monospace", color: COLORS.textMuted, marginTop: "2px" }}>
                    <span>0% 原图</span><span>100% 标准</span><span>150% 加强</span>
                  </div>
                </div>
              </div>
            )}

            {/* Tabs */}
            <div style={{ display: "flex", borderBottom: `1px solid ${COLORS.border}`, marginBottom: "28px" }}>
              {["report", "params", "xmp"].map(tab => (
                <button key={tab} onClick={() => setActiveTab(tab)} style={{
                  background: "none", border: "none",
                  borderBottom: activeTab === tab ? `1px solid ${COLORS.accent}` : "1px solid transparent",
                  color:        activeTab === tab ? COLORS.accent : COLORS.textMuted,
                  padding: "10px 24px", fontSize: "11px", letterSpacing: "0.12em",
                  fontFamily: "monospace", cursor: "pointer", marginBottom: "-1px",
                }}>
                  {{ report: "分析报告", params: "参数详情", xmp: "XMP预览" }[tab]}
                </button>
              ))}
              <div style={{ marginLeft: "auto", paddingBottom: "4px", display: "flex", gap: "8px" }}>
                {result.lut_content && (
                  <button onClick={downloadLut} style={{
                    background: COLORS.accent, border: `1px solid ${COLORS.accent}`,
                    color: "#000", padding: "6px 20px",
                    fontSize: "11px", letterSpacing: "0.1em",
                    fontFamily: "monospace", cursor: "pointer",
                  }}>
                    ↓ 下载 .cube LUT
                  </button>
                )}
                <button onClick={downloadXmp} style={{
                  background: "none", border: `1px solid ${COLORS.accentDim}`,
                  color: COLORS.accentDim, padding: "6px 16px",
                  fontSize: "11px", letterSpacing: "0.1em",
                  fontFamily: "monospace", cursor: "pointer",
                }}>
                  ↓ .xmp
                </button>
              </div>
            </div>

            {activeTab === "report" && <ReportTab result={result} />}
            {activeTab === "params" && <ParamsTab summary={result.summary} />}
            {activeTab === "xmp"    && <XmpTab xmp={result.xmp_content} />}
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

function FieldLabel({ children }) {
  return (
    <div style={{ fontSize: "9px", letterSpacing: "0.16em", color: COLORS.textMuted, fontFamily: "monospace", marginBottom: "5px" }}>
      {children}
    </div>
  );
}

function ReportTab({ result }) {
  return (
    <div style={{ background: COLORS.surface, border: `1px solid ${COLORS.border}`, padding: "24px" }}>
      <Label>分析摘要</Label>
      <Row label="分析模式"  value={result.mode === "B_dual" ? "双图 CNN 预测" : "单图特征提取"} />
      {result.cnn_used         && <Row label="CNN 参数"  value="已应用（R² 0.73）" good />}
      {result.compression_detected && <Row label="图片质量" value="检测到压缩，已补偿" warn />}
      {result.is_raw_source          && <Row label="原图格式" value="RAW（高精度）" good />}
      {result.curve_style && <Row label="曲线风格" value={result.curve_style} />}
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

function XmpTab({ xmp }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(xmp);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "8px" }}>
        <button onClick={copy} style={{
          background: "none", border: `1px solid ${COLORS.border}`,
          color: COLORS.textMuted, padding: "5px 14px",
          fontSize: "10px", fontFamily: "monospace", cursor: "pointer", letterSpacing: "0.06em",
        }}>
          {copied ? "已复制" : "复制"}
        </button>
      </div>
      <pre style={{
        background: COLORS.surface, border: `1px solid ${COLORS.border}`,
        padding: "20px", fontSize: "10px", fontFamily: "monospace",
        color: "#888", overflowX: "auto", overflowY: "auto",
        maxHeight: "500px", lineHeight: "1.6", margin: 0,
        whiteSpace: "pre-wrap", wordBreak: "break-all",
      }}>
        {xmp}
      </pre>
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
