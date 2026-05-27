import { useState, useCallback, useRef } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

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

const SCENE_OPTIONS = [
  { value: "auto",         label: "自动识别" },
  { value: "portrait",     label: "人像" },
  { value: "landscape",    label: "风光" },
  { value: "architecture", label: "建筑 / 城市" },
  { value: "still_life",   label: "静物" },
  { value: "night",        label: "夜景" },
  { value: "other",        label: "其他" },
];

const CAMERA_OPTIONS = [
  { value: "",          label: "不指定" },
  { value: "canon",     label: "Canon" },
  { value: "nikon",     label: "Nikon" },
  { value: "sony",      label: "Sony" },
  { value: "fuji",      label: "Fujifilm" },
  { value: "panasonic", label: "Panasonic" },
  { value: "olympus",   label: "Olympus" },
  { value: "leica",     label: "Leica" },
];

export default function App() {
  const [refFile,     setRefFile]     = useState(null);
  const [srcFile,     setSrcFile]     = useState(null);
  const [refPreview,  setRefPreview]  = useState(null);
  const [srcPreview,  setSrcPreview]  = useState(null);
  const [presetName,  setPresetName]  = useState("AI生成预设");
  const [refSceneType, setRefSceneType] = useState("auto");
  const [srcSceneType, setSrcSceneType] = useState("auto");
  const [cameraBrand, setCameraBrand] = useState("");
  const [loading,     setLoading]     = useState(false);
  const [result,      setResult]      = useState(null);
  const [error,       setError]       = useState(null);
  const [activeTab,   setActiveTab]   = useState("report");

  const refInputRef = useRef();
  const srcInputRef = useRef();

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
    form.append("ref_scene_type", refSceneType);
    form.append("src_scene_type", srcSceneType);
    form.append("camera_brand", cameraBrand);

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

  const downloadXmp = () => {
    if (!result?.xmp_content) return;
    const blob = new Blob([result.xmp_content], { type: "application/xml" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `${presetName}.xmp`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ minHeight: "100vh", background: COLORS.bg, color: COLORS.text, fontFamily: "'Georgia','Times New Roman',serif" }}>

      {/* Header */}
      <header style={{ borderBottom: `1px solid ${COLORS.border}`, padding: "28px 48px", display: "flex", alignItems: "baseline", gap: "16px" }}>
        <h1 style={{ margin: 0, fontSize: "22px", fontWeight: "400", letterSpacing: "0.12em", color: COLORS.accent, fontFamily: "'Georgia',serif" }}>
          PRESET FORGE
        </h1>
        <span style={{ color: COLORS.textMuted, fontSize: "12px", letterSpacing: "0.08em", fontFamily: "monospace" }}>
          Lightroom XMP Generator
        </span>
      </header>

      <main style={{ maxWidth: "1100px", margin: "0 auto", padding: "48px" }}>

        {/* Upload */}
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
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr auto", gap: "10px", alignItems: "center", marginBottom: "10px" }}>

            {/* Ref scene type */}
            <div>
              <FieldLabel>参考图场景</FieldLabel>
              <Select value={refSceneType} onChange={e => setRefSceneType(e.target.value)} options={SCENE_OPTIONS} />
            </div>

            {/* Src scene type */}
            <div>
              <FieldLabel>原图场景</FieldLabel>
              <Select value={srcSceneType} onChange={e => setSrcSceneType(e.target.value)} options={SCENE_OPTIONS} />
            </div>

            {/* Camera brand */}
            <div>
              <FieldLabel>相机品牌</FieldLabel>
              <Select value={cameraBrand} onChange={e => setCameraBrand(e.target.value)} options={CAMERA_OPTIONS} />
            </div>

            {/* Preset name */}
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

            {/* Analyze button */}
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
              {srcFile ? "● 双图模式 — 差值分析，精度更高" : "● 单图模式 — 风格特征提取"}
              {cameraBrand && <span style={{ marginLeft: "16px", color: COLORS.accentDim }}>
                ● 相机补偿：{CAMERA_OPTIONS.find(o => o.value === cameraBrand)?.label}
              </span>}
              {refSceneType !== "auto" && <span style={{ marginLeft: "16px", color: COLORS.accentDim }}>● 参考图：{SCENE_OPTIONS.find(o => o.value === refSceneType)?.label}</span>}
              {srcSceneType !== "auto" && <span style={{ marginLeft: "16px", color: COLORS.accentDim }}>● 原图：{SCENE_OPTIONS.find(o => o.value === srcSceneType)?.label}</span>}
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
              <div style={{ marginLeft: "auto", paddingBottom: "4px" }}>
                <button onClick={downloadXmp} style={{
                  background: "none", border: `1px solid ${COLORS.accent}`,
                  color: COLORS.accent, padding: "6px 20px",
                  fontSize: "11px", letterSpacing: "0.1em",
                  fontFamily: "monospace", cursor: "pointer",
                }}>
                  ↓ 下载 .xmp
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

function Select({ value, onChange, options }) {
  return (
    <select value={value} onChange={onChange} style={{
      width: "100%", background: COLORS.surface,
      border: `1px solid ${COLORS.border}`, color: COLORS.text,
      padding: "8px 12px", fontSize: "12px", fontFamily: "monospace",
      outline: "none", cursor: "pointer",
    }}>
      {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
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
  const report     = result.report || {};
  const confidence = report.confidence || {};

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}>
      <div style={{ background: COLORS.surface, border: `1px solid ${COLORS.border}`, padding: "24px" }}>
        <Label>风格识别</Label>
        <div style={{ fontSize: "18px", color: COLORS.accent, fontFamily: "'Georgia',serif", margin: "8px 0 12px" }}>
          {report.style_summary || "—"}
        </div>
        {report.style_keywords?.length > 0 && (
          <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginBottom: "12px" }}>
            {report.style_keywords.map(kw => (
              <span key={kw} style={{
                background: "#1a1a1a", border: `1px solid ${COLORS.border}`,
                padding: "3px 10px", fontSize: "10px", fontFamily: "monospace",
                color: COLORS.textDim, letterSpacing: "0.06em",
              }}>{kw}</span>
            ))}
          </div>
        )}
        <Row label="参考图场景" value={report.scene_label || report.scene_type} />
        {report.src_scene_label && <Row label="原图场景" value={report.src_scene_label} />}
        <Row label="光线环境"  value={report.lighting || "—"} />
        <Row label="分析模式"  value={result.mode === "B_dual" ? "双图差值" : "单图特征"} />
        {result.compression_detected && <Row label="图片质量" value="检测到压缩，已补偿" warn />}
        {result.is_raw_source          && <Row label="原图格式" value="RAW（高精度）" good />}
        {result.camera_note            && <Row label="相机补偿" value={result.camera_note} />}
        {result.curve_style            && <Row label="曲线风格" value={result.curve_style} />}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        <div style={{ background: COLORS.surface, border: `1px solid ${COLORS.border}`, padding: "24px" }}>
          <Label>参数置信度</Label>
          <div style={{ marginTop: "12px" }}>
            {Object.entries(confidence).filter(([k]) => k !== "overall").map(([key, val]) => (
              <ConfBar key={key} label={{
                tone_curve:        "色调曲线",
                hsl:               "HSL调整",
                color_grading:     "颜色分级",
                white_balance:     "白平衡",
                scene_recognition: "场景识别",
              }[key] || key} value={val} />
            ))}
          </div>
        </div>

        {report.warnings?.length > 0 && (
          <div style={{ background: "#110e08", border: "1px solid #3a2e18", padding: "20px" }}>
            <Label>注意事项</Label>
            <div style={{ marginTop: "10px" }}>
              {report.warnings.map((w, i) => (
                <div key={i} style={{ fontSize: "12px", color: "#c8a060", marginBottom: "8px", lineHeight: "1.6", fontFamily: "monospace" }}>
                  {w}
                </div>
              ))}
            </div>
          </div>
        )}
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

function ConfBar({ label, value }) {
  const pct   = Math.round(value * 100);
  const color = pct >= 75 ? COLORS.success : pct >= 50 ? COLORS.accent : COLORS.warning;
  return (
    <div style={{ marginBottom: "10px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px", fontSize: "11px", fontFamily: "monospace" }}>
        <span style={{ color: COLORS.textMuted }}>{label}</span>
        <span style={{ color }}>{pct}%</span>
      </div>
      <div style={{ height: "2px", background: COLORS.border }}>
        <div style={{ height: "100%", width: `${pct}%`, background: color, transition: "width 0.6s ease" }} />
      </div>
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
