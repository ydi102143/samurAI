// app/static/js/api.js
// =========================================
// samurAI Frontend API helper (unified)
// =========================================
const BASE = "/v1/datasets";
export const REGION = (window.__SAMURAI__?.region) || "any";

// --------------- utils -------------------
const bust = () => `&_=${Date.now()}`;

async function postJSON(url, body){
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {})
  });
  if(!r.ok) {
    let detail = "";
    try { detail = (await r.json())?.detail || ""; } catch {}
    throw new Error(`${url} failed${detail ? `: ${detail}` : ""}`);
  }
  return r.json();
}

// --------------- meta --------------------
export async function fetchTask(problem){
  const r = await fetch(`/v1/region/task?problem=${encodeURIComponent(problem)}`);
  if(!r.ok) throw new Error("task meta failed");
  return r.json();
}

// ----------- EDA schema/cols -------------
export async function fetchSchema(problem){
  const r = await fetch(`${BASE}/${REGION}/eda/schema?problem=${encodeURIComponent(problem)}`);
  if(!r.ok) throw new Error("schema failed");
  return r.json();
}

export async function fetchCols(problem){
  const r = await fetch(`${BASE}/${REGION}/preproc/columns/candidates?problem=${encodeURIComponent(problem)}`);
  if(!r.ok) throw new Error("columns failed");
  const js = await r.json();
  // サーバが numeric/categorical でも num/cat でも吸収
  const numeric = js.num ?? js.numeric ?? [];
  const categorical = js.cat ?? js.categorical ?? [];
  const drop = new Set(["date"]); // 念のため弾く
  const num = (numeric || []).filter(c => !drop.has(c));
  const cat = (categorical || []).filter(c => !drop.has(c));
  const all = (js.all ?? [...num, ...cat]).filter(c => !drop.has(c));
  return { num, cat, all };
}

// --------------- EDA PNG -----------------
export const fetchMissingPNG    = (p)=> `${BASE}/${REGION}/eda/plot/missing?problem=${encodeURIComponent(p)}${bust()}`;
export const fetchCorrPNG       = (p)=> `${BASE}/${REGION}/eda/plot/corr?problem=${encodeURIComponent(p)}${bust()}`;
export const fetchUnivarPNG     = (p,c)=> `${BASE}/${REGION}/eda/plot/univar?problem=${encodeURIComponent(p)}&col=${encodeURIComponent(c)}${bust()}`;
export const fetchScatterPNG    = (p,x,y)=> `${BASE}/${REGION}/eda/plot/scatter?problem=${encodeURIComponent(p)}&x=${encodeURIComponent(x)}&y=${encodeURIComponent(y)}${bust()}`;

// ------------- Model catalog/params ------
export async function fetchModelCatalog(region, problem){
  const r = await fetch(`${BASE}/${encodeURIComponent(region)}/models/catalog?problem=${encodeURIComponent(problem)}`);
  if(!r.ok) throw new Error("catalog failed");
  return r.json();
}
export async function fetchModelParams(region, model){
  const r = await fetch(`${BASE}/${encodeURIComponent(region)}/models/params?model=${encodeURIComponent(model)}`);
  if(!r.ok) throw new Error("params failed");
  return r.json();
}

// ---------------- Preprocess -------------
export const saveImpute = (p, { col, strategy, constant }) =>
  postJSON(
    `${BASE}/${REGION}/preproc/missing/save?problem=${encodeURIComponent(p)}`,
    { rules: [{ col, strategy, ...(strategy === "constant" ? { value: constant } : {}) }] }
  );

export const saveImputeAll = (p, { strategy, constant }) => {
  const q = new URLSearchParams({ problem: p, strategy });
  if (strategy === "constant") q.set("value", constant ?? "");
  return postJSON(`${BASE}/${REGION}/preproc/missing/save?${q.toString()}`, {});
};

export const saveOnehot = (p, { cols }) =>
  postJSON(
    `${BASE}/${REGION}/preproc/encode/save?problem=${encodeURIComponent(p)}&columns=${encodeURIComponent((cols||[]).join(","))}`,
    {}
  );

// /preproc/scale/save がない環境に備えて /preproc/scale にフォールバック
export async function saveScale(p, { kind, cols }){
  const qs = new URLSearchParams({
    problem: p,
    kind: String(kind || "")
  });
  const body = { columns: cols || [] };
  const url1 = `${BASE}/${REGION}/preproc/scale/save?${qs}`;
  try {
    return await postJSON(url1, {});
  } catch (_e1) {
    const url2 = `${BASE}/${REGION}/preproc/scale?${qs}`;
    return await postJSON(url2, body);
  }
}

export const saveFeatureSelection = (p, { num, cat }) =>
  postJSON(
    `${BASE}/${REGION}/preproc/feature/save?problem=${encodeURIComponent(p)}`,
    { num_cols: num || [], cat_cols: cat || [] }
  );

export const resetPreproc = (p) =>
  postJSON(`${BASE}/${REGION}/preproc/reset?problem=${encodeURIComponent(p)}`, {});

// ----------------- Train -----------------
/** 推奨：region/problem/model/params を明示して学習 */
export function postTrain(region, problem, model, params){
  return postJSON(
    `${BASE}/${encodeURIComponent(region)}/train?problem=${encodeURIComponent(problem)}`,
    { model, params: params || {} }
  );
}

/** 互換：旧署名 trainModel(problem, body:{model,params}) */
export function trainModel(problem, body){
  const model = body?.model;
  const params = body?.params || {};
  if(!model) throw new Error("trainModel: body.model が必要です");
  return postTrain(REGION, problem, model, params);
}

// ---------------- Evaluate ----------------
/** サーバ実装：GET /{region}/evaluate?problem=...&model=... */
export async function evalModel(region, problem, model){
  const r = await fetch(
    `${BASE}/${encodeURIComponent(region)}/evaluate?problem=${encodeURIComponent(problem)}&model=${encodeURIComponent(model)}`
  );
  if(!r.ok) throw new Error("evaluate failed");
  return r.json();
}

/** 互換：旧 evaluateOnce(problem, body:{model}) → 内部で GET /evaluate に変換 */
export function evaluateOnce(problem, body){
  const model = (body && body.model) || (typeof body === "string" ? body : null);
  if(!model) throw new Error("evaluateOnce: model がありません");
  return evalModel(REGION, problem, model);
}

/** 互換：旧 evaluateCV(problem, body) — 現状 once と同じ動作に寄せる */
export function evaluateCV(problem, body){
  return evaluateOnce(problem, body);
}

// -------------- Evaluation PNG ------------
/** 新：推奨の画像URL */
export const confPNG = (region, problem, model) =>
  `${BASE}/${encodeURIComponent(region)}/eval/cm.png?problem=${encodeURIComponent(problem)}&model=${encodeURIComponent(model)}${bust()}`;
export const rocPNG  = (region, problem, model) =>
  `${BASE}/${encodeURIComponent(region)}/eval/roc.png?problem=${encodeURIComponent(problem)}&model=${encodeURIComponent(model)}${bust()}`;
export const regPNG  = (region, problem, model) =>
  `${BASE}/${encodeURIComponent(region)}/eval/reg_scatter.png?problem=${encodeURIComponent(problem)}&model=${encodeURIComponent(model)}${bust()}`;

/** 互換：旧名（/eval/plot/* を使っていた呼び出しを吸収して新URLを返す） */
export const fetchConfusionPNG  = (p,m)=> confPNG(REGION, p, m);
export const fetchROCPNG        = (p,m)=> rocPNG(REGION, p, m);
export const fetchRegScatterPNG = (p,m)=> regPNG(REGION, p, m);

// --------------- Model Cards --------------
export async function fetchModelCards(region, problem){
  const r = await fetch(`${BASE}/${encodeURIComponent(region)}/cards/list?problem=${encodeURIComponent(problem)}`);
  if(!r.ok) throw new Error("cards failed");
  return r.json();
}
export function downloadCardsCSV(region, problem){
  const a = document.createElement("a");
  a.href = `${BASE}/${encodeURIComponent(region)}/cards.csv?problem=${encodeURIComponent(problem)}${bust()}`;
  a.download = `model_cards_${problem}.csv`;
  a.click();
}

// ---------------- Submit -------------------
export const submitResult = (p, body)=>
  postJSON(`${BASE}/${REGION}/submit?problem=${encodeURIComponent(p)}`, body);

// ---- default export（app.js が import api from "./api.js" で読めるように）----
const api = {
  BASE,
  REGION,
  fetchTask,
  fetchSchema,
  fetchCols,
  fetchMissingPNG,
  fetchCorrPNG,
  fetchUnivarPNG,
  fetchScatterPNG,
  fetchModelCatalog,
  fetchModelParams,
  saveImpute,
  saveImputeAll,
  saveOnehot,
  saveScale,
  saveFeatureSelection,
  resetPreproc,
  postTrain,
  trainModel,
  evalModel,
  evaluateOnce,
  evaluateCV,
  confPNG,
  rocPNG,
  regPNG,
  fetchConfusionPNG,
  fetchROCPNG,
  fetchRegScatterPNG,
  fetchModelCards,
  downloadCardsCSV,
  submitResult,
};
export default api;

// --- 互換エイリアス（呼び出し名の違いを吸収） ---
export const missingPNG = (p, src="raw") =>
  fetchMissingPNG(p).replace("plot/missing?", `plot/missing?source=${encodeURIComponent(src)}&`);

export const univarPNG  = (p, c)     => fetchUnivarPNG(p, c);
export const scatterPNG = (p, x, y)  => fetchScatterPNG(p, x, y);

// コリレーション: 旧 corrPNG({region, problem, ...}) を受けても動くように
export function corrPNG(problem, { region = REGION, method = "pearson", max_cols } = {}){
  const base = fetchCorrPNG(problem);
  const u = new URL(base, location.origin);
  if (region)   u.pathname = u.pathname.replace(`/${REGION}/`, `/${encodeURIComponent(region)}/`);
  u.searchParams.set("method", method);
  if (Number.isFinite(max_cols)) u.searchParams.set("max_cols", String(max_cols));
  return u.pathname + "?" + u.searchParams.toString();
}

// リセット/欠損PNG URL 互換
export const resetPipeline = (p) => postJSON(`${BASE}/${REGION}/preproc/reset?problem=${encodeURIComponent(p)}`, {});
export const getMissingPlot = (region, p, src="raw") =>
  `${BASE}/${encodeURIComponent(region)}/eda/plot/missing?problem=${encodeURIComponent(p)}&source=${encodeURIComponent(src)}&_=${Date.now()}`;
