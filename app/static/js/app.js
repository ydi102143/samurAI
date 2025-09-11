// app/static/js/app.js
import * as api from "./api.js";

// ---- globals (offline.html で埋め込まれる変数を使う) ----
const problem = (window.__SAMURAI__?.problem) || "";
const region  = (window.__SAMURAI__?.region)  || "any";

const $  = (sel, root=document)=>root.querySelector(sel);
const $$ = (sel, root=document)=>Array.from(root.querySelectorAll(sel));

// ===================== UI Bindings =====================
function bindTabs(){
  const btns = $$(".tab-btn");
  const tabs = $$(".tab");
  btns.forEach(b=>{
    b.onclick = ()=>{
      btns.forEach(x=>x.classList.remove("active"));
      tabs.forEach(t=>t.classList.remove("active"));
      b.classList.add("active");
      const pane = $("#tab-"+b.dataset.tab);
      if(pane) pane.classList.add("active");
    };
  });
}

// 親: .subtabs / 子: .subtab-btn(data-subtab=...) と .subtab(#subtab-*)
function bindSubtabs(){
  document.querySelectorAll(".subtabs .subtab-btn").forEach(btn=>{
    btn.addEventListener("click", ()=>{
      const wrap = btn.closest(".subtabs");
      wrap.querySelectorAll(".subtab-btn").forEach(b=>b.classList.remove("active"));
      btn.classList.add("active");

      const target = btn.dataset.subtab; // "basic" | "ensemble" / "bag" | "boost" | "stack"
      const sect = wrap.parentElement;
      sect.querySelectorAll(".subtab").forEach(p=>p.classList.remove("active"));
      const pane = sect.querySelector(`#subtab-${target}`) || sect.querySelector(`#sub-${target}`);
      if (pane) pane.classList.add("active");
    });
  });
}

// ===================== helpers =====================
function tableHTML(cols, rows){
  const th = cols.map(c=>`<th>${c}</th>`).join("");
  const tb = (rows||[]).map(r=>`<tr>${cols.map(c=>`<td>${r[c]??""}</td>`).join("")}</tr>`).join("") || `<tr><td>(no rows)</td></tr>`;
  return `<div class="peek-wrap"><table class="peek"><thead><tr>${th}</tr></thead><tbody>${tb}</tbody></table></div>`;
}
function setOptions(id, arr, {multiple=false, preselect=0}={}){
  const el = $("#"+id);
  if(!el) return;
  el.innerHTML = (arr||[]).map(v=>`<option value="${v}">${v}</option>`).join("");
  if(multiple && preselect>0){
    for(let i=0;i<el.options.length && i<preselect;i++) el.options[i].selected = true;
  }
}

// ====== 安全に特徴量セレクトを埋める（候補API→ダメならプレビュー推定） ======
async function populateFeatureSelectsSafe(problem){
  // 1) サーバの candidates API を第一候補
  try{
    const cols = await api.fetchCols(problem);
    if ((cols?.num?.length || 0) + (cols?.cat?.length || 0) > 0){
      setOptions("featNum", cols.num || [], {multiple:true, preselect:3});
      setOptions("featCat", cols.cat || [], {multiple:true, preselect:3});
      return;
    }
  }catch(_){ /* fallthrough */ }

  // 2) 取れなければ schema プレビューから型推定（先頭行ベース）
  try{
    const js = await api.fetchSchema(problem);
    const columns = js.preview?.columns || [];
    const rows    = js.preview?.rows || [];
    if (!columns.length) return;
    const sample = rows[0] || {};
    const num = [], cat = [];
    for (const c of columns){
      if (c === "y" || c === "date") continue; // 除外
      const v = sample[c];
      const looksNum = typeof v === "number" || (!isNaN(parseFloat(v)) && v !== "" && v !== null);
      (looksNum ? num : cat).push(c);
    }
    setOptions("featNum", num, {multiple:true, preselect:3});
    setOptions("featCat", cat, {multiple:true, preselect:3});
  }catch(_){ /* no-op */ }
}

// ===================== Meta / Task =====================
async function initMetaAndTask(){
  const tk = await api.fetchTask(problem);
  const info = `目的変数: ${tk.target?.name||"y"} / タイプ: ${tk.target?.type||"-"} / 指標: ${tk.metric||"-"} / レベル: ${tk.level??"-"}`;
  const area = $("#taskInfo") || $("#aboutOut");
  if(area) area.innerHTML = info;

  // Lv3〜の追加EDAブロック
  if((tk.level||1) >= 3){
    const edaLv3 = $("#edaLv3");
    if(edaLv3) edaLv3.style.display = "";
  }

  // 上位機能のロック
  const lv = tk.level || 1;
  const lock = (selector, needLv) => {
    const el = document.querySelector(selector);
    if (!el) return;
    if (lv < needLv) {
      el.classList.add("is-locked");
      el.querySelectorAll("button,select,input,textarea").forEach(x => x.disabled = true);
    } else {
      el.classList.remove("is-locked");
      el.querySelectorAll("button,select,input,textarea").forEach(x => x.disabled = false);
    }
  };
  lock("#sub-bag", 3);
  const subBoost = document.querySelector("#sub-boost");
  if (subBoost) subBoost.style.display = (lv >= 6) ? "" : "none";
  lock("#sub-boost", 6);
  const boostLv7 = document.querySelector("#boostLv7");
  if (boostLv7) boostLv7.style.display = (lv >= 7) ? "" : "none";
  lock("#boostLv7", 7);
  const subStack = document.querySelector("#sub-stack");
  if (subStack) subStack.style.display = (lv >= 9) ? "" : "none";
  lock("#sub-stack", 9);

  // Lv4〜でスケーリングブロックを表示
  const scaleBlock = document.querySelector("#scaleBlock");
  if (scaleBlock) scaleBlock.style.display = (lv >= 4) ? "" : "none";

  // メタ chips
  $("#chip-level") && ($("#chip-level").textContent = lv);
  $("#chip-metric") && ($("#chip-metric").textContent = tk.metric || "-");
}

// ===================== EDA =====================
async function initEDA(){
  // 20行プレビュー
  const btnSchema = $("#btnSchema");
  if(btnSchema){
    btnSchema.onclick = async ()=>{
      try{
        const js = await api.fetchSchema(problem);
        const cols = js.preview?.columns || [];
        const rows = js.preview?.rows || [];
        $("#schemaOut").innerHTML = tableHTML(cols, rows);

        // 型ざっくり推定→各種ドロップダウン更新
        const numeric = [];
        const categorical = [];
        if (rows.length && cols.length) {
          const sample = rows[0];
          for (const c of cols) {
            const v = sample[c];
            const isNum = typeof v === "number" || (!isNaN(parseFloat(v)) && v !== "" && v !== null);
            if (c === "date" || c === "y") continue;
            (isNum ? numeric : categorical).push(c);
          }
        }
        setOptions("imputeCol", [...numeric, ...categorical]);
        setOptions("onehotCols", categorical, {multiple:true, preselect:3});
        setOptions("scaleCols", numeric, {multiple:true, preselect:3});
        setOptions("univarCol", numeric);
        setOptions("scatterX", numeric);
        setOptions("scatterY", numeric);

        // 特徴量セレクトも同期
        await populateFeatureSelectsSafe(problem);

      }catch(e){
        $("#schemaOut").textContent = "失敗: " + e;
      }
    };
  }

  // 欠損率バー（preprocタブ左上の<img>）
  const missingImg = $("#eda-missing-img");
  if (missingImg && problem){
    missingImg.src = (typeof api.fetchMissingPNG==="function")
      ? api.fetchMissingPNG(problem)
      : `/v1/datasets/${encodeURIComponent(region)}/eda/plot/missing?problem=${encodeURIComponent(problem)}&_=${Date.now()}`;
  }

  // Lv3: 単変量/散布図/相関
  const btnUnivar = $("#btnUnivar");
  if(btnUnivar){
    btnUnivar.onclick = ()=>{
      const col = $("#univarCol")?.value || "";
      const el = $("#univarOut"); if(!el) return;
      const img = new Image();
      img.onload = ()=>{ el.innerHTML=""; el.appendChild(img); };
      img.src = (typeof api.fetchUnivarPNG==="function")
        ? api.fetchUnivarPNG(problem, col)
        : `/v1/datasets/${encodeURIComponent(region)}/eda/plot/univar?problem=${encodeURIComponent(problem)}&col=${encodeURIComponent(col)}&_=${Date.now()}`;
    };
  }

  const btnScatter = $("#btnScatter");
  if(btnScatter){
    btnScatter.onclick = ()=>{
      const x = $("#scatterX")?.value || "";
      const y = $("#scatterY")?.value || "";
      const el = $("#scatterOut"); if(!el) return;
      const img = new Image();
      img.onload = ()=>{ el.innerHTML=""; el.appendChild(img); };
      img.src = (typeof api.fetchScatterPNG==="function")
        ? api.fetchScatterPNG(problem, x, y)
        : `/v1/datasets/${encodeURIComponent(region)}/eda/plot/scatter?problem=${encodeURIComponent(problem)}&x=${encodeURIComponent(x)}&y=${encodeURIComponent(y)}&_=${Date.now()}`;
    };
  }

  const btnCorr = $("#btnCorr");
  if (btnCorr){
    btnCorr.onclick = ()=>{
      const imgEl = $("#corrOut"); if(!imgEl) return;
      const url = (typeof api.fetchCorrPNG==="function")
        ? api.fetchCorrPNG(problem)
        : `/v1/datasets/${encodeURIComponent(region)}/eda/plot/corr?problem=${encodeURIComponent(problem)}&_=${Date.now()}`;
      imgEl.alt = `Correlation heatmap`;
      imgEl.src = url;
    };
  }
}

// ===================== Preprocess =====================
async function initPreprocess(){
  // まず安全に特徴量セレクトを埋める
  await populateFeatureSelectsSafe(problem);

  // そのうえで他のセレクトも通常どおり
  const cols = await api.fetchCols(problem).catch(()=>({num:[],cat:[],all:[]}));
  setOptions("imputeCol", cols.all);
  setOptions("onehotCols", cols.cat, {multiple:true, preselect:3});
  setOptions("scaleCols", cols.num, {multiple:true, preselect:3});
  setOptions("univarCol", cols.num);
  setOptions("scatterX", cols.num);
  setOptions("scatterY", cols.num);

  // 欠損（列単位）
  const btnImpute = $("#btnImputeSave");
  if(btnImpute){
    btnImpute.onclick = async ()=>{
      const col = $("#imputeCol")?.value || "";
      const method = $("#imputeMethod")?.value || "mean";
      const v = $("#imputeConst")?.value || "";
      const body = { col, strategy:method, constant:v };
      $("#preprocMsg").textContent = "保存中…";
      try{
        await api.saveImpute(problem, body);
        $("#preprocMsg").textContent = "OK: processed を更新しました";
        const img = $("#eda-missing-img");
        if (img) img.src = api.fetchMissingPNG(problem);
        await populateFeatureSelectsSafe(problem); // 変化に追随
      }catch(e){
        $("#preprocMsg").textContent = "エラー: "+e;
      }
    };
  }

  // 欠損（一括）
  const btnImputeAll = $("#btnImputeAll");
  if(btnImputeAll){
    btnImputeAll.onclick = async ()=>{
      const method = $("#imputeAllMethod")?.value || "";
      const v = $("#imputeAllConst")?.value || "";
      if(!method){ $("#preprocMsg").textContent = "戦略を選んでください"; return; }
      $("#preprocMsg").textContent = "保存中…";
      try{
        await api.saveImputeAll(problem, { strategy:method, constant:v });
        $("#preprocMsg").textContent = "OK: processed を更新しました";
        const img = $("#eda-missing-img");
        if (img) img.src = api.fetchMissingPNG(problem);
        await populateFeatureSelectsSafe(problem);
      }catch(e){
        $("#preprocMsg").textContent = "エラー: "+e;
      }
    };
  }

  // One-Hot
  const btnOnehot = $("#btnOnehot");
  if(btnOnehot){
    btnOnehot.onclick = async ()=>{
      const el = $("#onehotCols");
      const colsSel = [...el.options].filter(o=>o.selected).map(o=>o.value);
      $("#preprocMsg").textContent = "保存中…";
      try{
        await api.saveOnehot(problem, { cols: colsSel });
        $("#preprocMsg").textContent = "OK: One-Hot を反映しました";
        await populateFeatureSelectsSafe(problem); // 列構成が変わるため再同期
      }catch(e){
        $("#preprocMsg").textContent = "エラー: "+e;
      }
    };
  }

  // スケーリング
  const btnScale = $("#btnScale");
  if(btnScale){
    btnScale.onclick = async ()=>{
      const kind = String($("#scaleKind")?.value || "standard");
      const el = $("#scaleCols");
      const cols = [...el.options].filter(o=>o.selected).map(o=>o.value);
      $("#preprocMsg").textContent = "保存中…";
      try{
        await api.saveScale(problem, { kind, cols });
        $("#preprocMsg").textContent = "OK: スケーリングを反映しました";
        await populateFeatureSelectsSafe(problem);
      }catch(e){
        $("#preprocMsg").textContent = "エラー: "+e;
      }
    };
  }

  // 特徴量選択
  const btnFeat = $("#btnFeatSave");
  if(btnFeat){
    btnFeat.onclick = async ()=>{
      const num = [...($("#featNum")?.options||[])].filter(o=>o.selected).map(o=>o.value);
      const cat = [...($("#featCat")?.options||[])].filter(o=>o.selected).map(o=>o.value);
      $("#preprocMsg").textContent = "保存中…";
      try{
        await api.saveFeatureSelection(problem, { num, cat });
        $("#preprocMsg").textContent = "OK: 特徴量選択を保存しました";
      }catch(e){
        $("#preprocMsg").textContent = "エラー: "+e;
      }
    };
  }

  // リセット
  const btnReset = $("#btnPreprocReset");
  if(btnReset){
    btnReset.onclick = async ()=>{
      $("#preprocMsg").textContent = "リセット中…";
      try{
        await api.resetPreproc(problem);
        $("#preprocMsg").textContent = "OK: 元に戻しました";
        const img = $("#eda-missing-img");
        if (img) img.src = api.fetchMissingPNG(problem).replace(/&_=\d+$/,'') + `&_=${Date.now()}`;
        await populateFeatureSelectsSafe(problem); // 元状態に同期
      }catch(e){
        $("#preprocMsg").textContent = "エラー: "+e;
      }
    };
  }
}

// ===================== Models =====================
async function buildModelCard(model){
  const card = document.createElement("div");
  card.className = "card model-card";
  card.innerHTML = `
    <div class="model-header">
      <h3>${model.label||model.name}</h3>
      <code>${model.name}</code>
    </div>
    <div class="model-body"><div class="params-grid"></div></div>
    <div class="model-footer"><button class="btn btn-train">学習</button> <span class="msg muted small"></span></div>
  `;

  // パラメータスキーマ
  let schema = {};
  try {
    const sch = await api.fetchModelParams(region, model.name);
    schema = sch?.schema || {};
  } catch { schema = {}; }

  // フォーム生成
  const grid = card.querySelector(".params-grid");
  for(const [k, spec] of Object.entries(schema)){
    const id = `p-${model.name}-${k}-${Math.random().toString(36).slice(2,7)}`;
    const wrap = document.createElement("div");
    wrap.className = "param";
    let input = "";
    if(spec.type==="select"){
      input = `<select id="${id}" data-key="${k}">
        ${(spec.choices||[]).map(v=>`<option ${v===spec.default?'selected':''} value="${v}">${String(v)}</option>`).join("")}
      </select>`;
    }else if(spec.type==="bool"){
      input = `<input id="${id}" data-key="${k}" type="checkbox" ${spec.default?'checked':''}>`;
    }else if(spec.type==="int"||spec.type==="float"){
      const min = spec.range?.[0] ?? ""; const max = spec.range?.[1] ?? ""; const step = spec.step ?? 1;
      const def = spec.default ?? "";
      input = `<input id="${id}" data-key="${k}" type="number" value="${def}" min="${min}" max="${max}" step="${step}">`;
    }else{
      input = `<input id="${id}" data-key="${k}" type="text" value="${spec.default ?? ""}">`;
    }
    wrap.innerHTML = `<label for="${id}">${k}</label>${input}${ spec.desc ? `<small>${spec.desc}</small>` : "" }`;
    grid.appendChild(wrap);
  }

  // 学習
  const btn = card.querySelector(".btn.btn-train");
  const msg = card.querySelector(".msg");
  btn.onclick = async ()=>{
    try{
      btn.disabled = true; msg.textContent = "学習中…";
      const params = {};
      grid.querySelectorAll("[data-key]").forEach(el=>{
        const k = el.dataset.key;
        if(el.type==="checkbox") params[k] = !!el.checked;
        else if(el.type==="number"){
          const isFloat = String(el.step||"").includes(".");
          params[k] = el.value==="" ? null : (isFloat ? parseFloat(el.value) : parseInt(el.value,10));
        }else params[k] = el.value;
      });
      await api.postTrain(region, problem, model.name, params);
      msg.textContent = "学習完了（保存済み）";

      const sel = $("#evalModel");
      if(sel && !Array.from(sel.options).some(o=>o.value===model.name)){
        const opt = document.createElement("option");
        opt.value = model.name; opt.textContent = model.label||model.name;
        sel.appendChild(opt);
      }
    }catch(e){
      msg.textContent = "エラー: " + (e?.message ?? String(e));
    }finally{ btn.disabled = false; }
  };

  return card;
}

async function initModelsUI(){
  // カタログ（短名）
  let cat = {};
  try { cat = await api.fetchModelCatalog(region, problem); } catch { cat = {}; }

  const blocks = [
    ["#cards-basic",  cat.basic||[]],
    ["#cards-bag",    cat.bagging||[]],
    ["#cards-boost",  cat.boosting||[]],
    ["#cards-xgb",    cat.xgb||[]],
    ["#cards-lgbm",   cat.lgbm||[]],
    ["#cards-cat",    cat.cat||[]],
    ["#cards-stack",  cat.stacking||[]],
  ];
  for (const [sel, list] of blocks){
    const box = document.querySelector(sel);
    if(!box) continue;
    box.innerHTML = "";
    if(!list.length){ box.innerHTML = `<div class="muted small">(なし)</div>`; continue; }
    const cards = await Promise.all(list.map(buildModelCard));
    cards.forEach(c => box.appendChild(c));
  }

  // 評価セレクト
  const evalSel = $("#evalModel");
  if(evalSel){
    evalSel.innerHTML = "";
    [...(cat.basic||[]), ...(cat.bagging||[]), ...(cat.boosting||[]), ...(cat.stacking||[])]
      .forEach(m=>{
        const opt = document.createElement("option");
        opt.value = m.name; opt.textContent = m.label||m.name;
        evalSel.appendChild(opt);
      });

    // 評価ボタン
    const btnEval = $("#btnEvaluate");
    if(btnEval){
      btnEval.onclick = async ()=>{
        const m = evalSel.value;
        $("#evalMsg").textContent = "評価中…";
        try{
          const js = await api.evalModel(region, problem, m);
          $("#evalMsg").textContent =
            `score=${(js.score??0).toFixed(5)} (${js.metric}) / passed=${js.passed?"YES":"NO"}`;

          // 画像
          const ts = `&_=${Date.now()}`;
          const cmURL  = `/v1/datasets/${encodeURIComponent(region)}/eval/cm.png?problem=${encodeURIComponent(problem)}&model=${encodeURIComponent(m)}${ts}`;
          const rocURL = `/v1/datasets/${encodeURIComponent(region)}/eval/roc.png?problem=${encodeURIComponent(problem)}&model=${encodeURIComponent(m)}${ts}`;
          const regURL = `/v1/datasets/${encodeURIComponent(region)}/eval/reg_scatter.png?problem=${encodeURIComponent(problem)}&model=${encodeURIComponent(m)}${ts}`;

          const conf = $("#plotConf"); if(conf) conf.src = cmURL;
          const roc  = $("#plotROC");  if(roc)  roc.src = rocURL;
          const reg  = $("#plotREG");  if(reg)  reg.src = regURL;
        }catch(e){
          $("#evalMsg").textContent = "エラー: "+(e?.message ?? String(e));
        }
      };
    }
  }
}

// ===================== Submit =====================
async function initSubmit(){
  const btn = document.getElementById("btnSubmit");
  const summary = document.getElementById("lastEvalSummary");
  if(!btn) return;
  btn.onclick = async ()=>{
    // 評価セレクタから選択中モデルを使う
    const msel = document.getElementById("evalModel");
    const model = msel?.value;
    const out = document.getElementById("submitMsg");
    if(!model){
      if(out) out.textContent = "モデルが選択されていません";
      return;
    }
    if(out) out.textContent = "提出中…";
    try{
      const q = new URLSearchParams({ problem, model });
      const r = await fetch(`/v1/datasets/${encodeURIComponent(region)}/submit?${q.toString()}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}"
      });
      const js = await r.json();
      if(!r.ok) throw new Error(js?.detail || "submit failed");
      if(summary){
        summary.textContent = `model=${model} / score=${(js.score??0).toFixed(5)} (${js.metric}) / passed=${js.passed?"YES":"NO"}`;
      }
      if(out) out.textContent = js.submitted ? "提出完了（合格）" : "提出は保存しました（不合格）";
    }catch(e){
      if(out) out.textContent = "エラー: " + e;
    }
  };
}

// ===================== boot =====================
async function boot(){
  if(!problem){
    console.warn("problem が未指定です。テンプレートで window.__SAMURAI__.problem を入れてください。");
  }
  bindTabs();
  bindSubtabs();
  try{
    await initMetaAndTask();
    await initEDA();
    await initPreprocess();
    await initModelsUI();
    await initSubmit();
  }catch(e){
    console.error(e);
  }
}

document.addEventListener("DOMContentLoaded", boot);
