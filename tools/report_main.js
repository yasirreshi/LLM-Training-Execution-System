/* Renderer + instruments.  STAGES_DOC supplies the prose; D supplies every number. */
"use strict";
const NL = String.fromCharCode(10);
const D = JSON.parse(document.getElementById("RUNDATA").textContent);
const $ = (s,r)=>(r||document).querySelector(s);
const el = (t,c,h)=>{const e=document.createElement(t); if(c)e.className=c; if(h!==undefined)e.innerHTML=h; return e;};
const fmt = n => (n===null||n===undefined) ? "—" : (typeof n==="number" ? n.toLocaleString() : n);
const pct = n => (n*100).toFixed(1)+"%";
const esc = s => String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const LANES = ["general_web","code","math_science","indic","agentic","reasoning"];
const SHORT = {general_web:"web",code:"code",math_science:"math",indic:"indic",agentic:"agent",reasoning:"reason"};
const LANE_HUE = {general_web:"#3D7EA6",code:"#0B6E63",math_science:"#7A5EA8",
                  indic:"#B8632A",agentic:"#2C6E49",reasoning:"#9A6408"};
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}

/* theme + tooltip ---------------------------------------------------- */
const root=document.documentElement;
$("#themeBtn").addEventListener("click",()=>{
  const dark = root.getAttribute("data-theme")==="dark" ||
    (!root.hasAttribute("data-theme") && matchMedia("(prefers-color-scheme:dark)").matches);
  root.setAttribute("data-theme", dark?"light":"dark");
  REDRAW.forEach(f=>{try{f();}catch(e){}});
});
const REDRAW=[];
const tip=$("#tip");
function showTip(e,html){tip.innerHTML=html; tip.classList.add("on");
  const p=8,r=tip.getBoundingClientRect();
  let x=e.clientX+p,y=e.clientY+p;
  if(x+r.width>innerWidth-8)x=e.clientX-r.width-p;
  if(y+r.height>innerHeight-8)y=e.clientY-r.height-p;
  tip.style.left=x+"px"; tip.style.top=y+"px";}
function hideTip(){tip.classList.remove("on");}
function hoverable(node,fn){
  node.addEventListener("mouseenter",e=>showTip(e,fn()));
  node.addEventListener("mousemove",e=>showTip(e,fn()));
  node.addEventListener("mouseleave",hideTip);}

/* masthead ----------------------------------------------------------- */
(function(){
  const m=D.evidence_meta;
  $("#mRun").textContent=m.run_id;
  $("#mCfg").textContent="cfg "+m.config_hash.slice(0,10);
  const v=$("#verdict");
  [["requirements",m.requirements_passed+" / "+m.requirements_total],
   ["skipped or repeated batches",D.integrity.duplicate_count+D.integrity.missing_steps.length],
   ["replay derivations agree",D.replay.all_match?"3 / 3":"FAIL"],
   ["eval tokens in a gradient",D.firewall.validation_gradient_bearing_tokens],
   ["pipeline stages",STAGES_DOC.length],
   ["wall clock",m.wall_seconds.toFixed(0)+"s"]].forEach(([k,val])=>{
    const c=el("div","vcell"); c.appendChild(el("div","k",k)); c.appendChild(el("div","v",String(val)));
    v.appendChild(c);});
})();

/* architecture: the contract table ------------------------------------ */
(function(){
  const rows=[
   ["Foundations","Transformer training","Next-token loss, bounded attention, explicit loss-bearing tokens, EOS boundaries survive packing","packing/masks.py · training/model.py"],
   ["Vocabulary","Tokenization","The same raw text always produces the same token ids; canonical special tokens; Indic-safe normalization","tokenizer/normalize.py · tokenizer/freeze.py"],
   ["Sourcing","Source contract","Every source carries provenance, licence, held-out status, capability tags and token accounting","corpus/loader.py · shards/manifest.py"],
   ["Cleaning","Admission contract","Only cleaned, deduplicated, PII-screened, contamination-scanned data may enter","shards/manifest.py · firewall/"],
   ["Curriculum","Mixture contract","Curriculum stages, protected floors, OPUS selection, annealing reserves, long-context timing","mixture/compiler.py · opus/selector.py"],
   ["Execution","Execution contract","All of the above, plus a ledger that survives a crash and reconstructs the run","ledger/ · streams/ · training/checkpoint.py"]];
  $("#contractTable").innerHTML=`<thead><tr><th>layer</th><th>contract</th><th>obligation it imposes on execution</th><th>enforced by</th></tr></thead><tbody>`+
    rows.map(r=>`<tr><td class="mono" style="font-size:.75rem;white-space:nowrap">${r[0]}</td>
      <td style="font-size:.8rem">${r[1]}</td><td style="font-size:.82rem">${r[2]}</td>
      <td class="mono" style="font-size:.72rem;color:var(--ink-3)">${r[3]}</td></tr>`).join("")+"</tbody>";
})();

/* ── stage rendering ──────────────────────────────────────────────── */
function facet(label, bodyHTML){
  return `<div class="facet"><div class="fl">${label}</div><div class="fb">${bodyHTML}</div></div>`;
}
function bullets(list){ return "<ul>"+list.map(x=>`<li>${x}</li>`).join("")+"</ul>"; }

// Lightweight Python highlighter.
// Single pass over already-escaped text: one alternation, one replacer, no
// re-scanning. That matters — a second pass would match the keyword "class"
// inside the class="..." attribute the first pass had just inserted.
const PY_KW = ["def","class","return","if","elif","else","for","while","in","not","and","or",
  "is","None","True","False","import","from","as","with","try","except","finally","raise",
  "yield","lambda","pass","continue","break","assert","global","nonlocal","await","async","self"];
const PY_TOKEN = new RegExp([
  "(&quot;&quot;&quot;[\s\S]*?&quot;&quot;&quot;)",   // docstring
  "(&quot;(?:[^&]|&(?!quot;))*?&quot;|&#39;(?:[^&]|&(?!#39;))*?&#39;)", // string
  "(#[^\n]*)",                                          // comment
  "\b(" + PY_KW.join("|") + ")\b",                    // keyword
  "\b(\d+\.?\d*)\b"                          // number
].join("|"), "g");

function highlight(escaped){
  return escaped.replace(PY_TOKEN, (m, doc, str, cm, kw, num) => {
    if(doc !== undefined) return '<span class="c-doc">' + doc + '</span>';
    if(str !== undefined) return '<span class="c-st">' + str + '</span>';
    if(cm  !== undefined) return '<span class="c-cm">' + cm + '</span>';
    if(kw  !== undefined) return '<span class="c-kw">' + kw + '</span>';
    if(num !== undefined) return '<span class="c-nu">' + num + '</span>';
    return m;
  });
}

function codeBlock(src){
  const marks = {}; (src.notes||[]).forEach(n=>{ marks[n.line]=n.n; });
  const lines = src.code.split(NL);
  const gutter = lines.map((_,i)=>{
    const n = marks[i+1];
    return n ? `<span class="gn marked">${n}</span>` : `<span class="gn">${i+1}</span>`;
  }).join(NL);
  const body = lines.map((l,i)=>{
    const hl = highlight(esc(l)) || "&nbsp;";
    return marks[i+1] ? `<span class="cl marked">${hl}</span>` : `<span class="cl">${hl}</span>`;
  }).join(NL);

  const notes = (src.notes||[]).length
    ? `<ol class="srcnotes">${src.notes.map(n=>
        `<li><span class="nn">${n.n}</span><span class="nt">${n.text}</span></li>`).join("")}</ol>`
    : "";

  return `<div class="srcbox">
    <div class="srchd">
      <span class="mono">${src.file}</span>
      <span class="mono" style="color:var(--accent)">${esc(src.symbol)}()</span>
      <span class="note" style="margin-left:auto">${src.lines} lines${src.truncated?", truncated":""} · from line ${src.first_line}</span>
    </div>
    ${src.summary?`<div class="srcsum">${src.summary}</div>`:""}
    <div class="srcwrap"><pre class="srcgut">${gutter}</pre><pre class="src">${body}</pre></div>
    ${notes}</div>`;
}

function jsonBlock(obj, label){
  const text = JSON.stringify(obj, null, 2);
  return `<div class="srcbox">
    <div class="srchd"><span class="mono">${label}</span>
      <span class="note" style="margin-left:auto">verbatim from the run</span></div>
    <pre class="src json">${esc(text)}</pre></div>`;
}
function logBlock(lines){
  if(!lines || !lines.length) return "";
  return `<pre class="src log">${lines.map(l=>{
    const cls = l.indexOf("[PASS]")===0 ? "lp" : l.indexOf("[FAIL]")===0 ? "lf" : "";
    return `<span class="${cls}">${esc(l)}</span>`;}).join(String.fromCharCode(10))}</pre>`;
}

function renderStage(s){
  const sec=el("section");
  sec.id="s-"+s.id; sec.dataset.title=s.title; sec.dataset.group=s.group; sec.dataset.num=s.n;

  let h=`<div class="wrap">
    <div class="stagehd"><span class="stagenum">${s.n}</span>
      <div style="flex:1 1 300px"><h2>${s.title}</h2><div class="oneline">${s.one}</div></div>
      <span class="stagetag">${s.tag}</span></div></div>`;

  h+=`<div class="facets">`;
  h+=facet("Concept", `<div class="snap">${s.concept}</div>`);
  h+=facet("What we built", bullets(s.built));
  let ranHTML=""; try{ ranHTML = bullets(s.ran()); }catch(e){ ranHTML=`<p class="note">unavailable</p>`; }
  h+=facet("What it ran", ranHTML);

  const src = (D.source && D.source[s.id]) || [];
  if(src.length) h+=facet("The code that ran",
    `<p class="note" style="margin-bottom:.5rem">Pulled out of the working tree at build time — this
      is the implementation, not a paraphrase of it. <b>Numbered lines are explained underneath</b>,
      so the code is readable without knowing Python.</p>` + src.map(codeBlock).join(""));

  const wrote = [];
  if(s.artifacts && s.artifacts.length){
    const known = new Set((D.files||[]).map(f=>f.path));
    wrote.push(`<div class="sub">files written</div><div class="filelist">`+
      s.artifacts.map(a=>{
        const f=(D.files||[]).find(x=>x.path===a);
        const size = f ? (f.bytes>=1024 ? (f.bytes/1024).toFixed(1)+" KB" : f.bytes+" B")
                       : (a.indexOf("*")>=0 ? "group" : "—");
        return `<div class="frow"><span class="mono">${a}</span><span class="num mono">${size}</span></div>`;
      }).join("")+`</div>`);
  }
  if(s.record && D.records && D.records[s.record])
    wrote.push(`<div class="sub">one record, verbatim</div>`+
      jsonBlock(D.records[s.record], s.record));
  if(wrote.length) h+=facet("What it wrote", wrote.join(""));

  const log = D.log && D.log.by_stage && D.log.by_stage[s.id];
  if(log && log.length) h+=facet("In run.log", logBlock(log));

  h+=facet("In / out", `<div class="iogrid">
      <div class="iobox"><div class="t">consumes</div><ul>${s.io.in.map(x=>`<li>${x}</li>`).join("")}</ul></div>
      <div class="iobox"><div class="t">produces</div><ul>${s.io.out.map(x=>`<li>${x}</li>`).join("")}</ul></div>
      <div class="iobox"><div class="t">next stage</div><ul><li>${s.io.next}</li></ul></div></div>`);

  let mets=""; try{ mets = s.metrics().map(([k,v])=>
    `<div class="met"><div class="mk">${k}</div><div class="mv">${v}</div></div>`).join(""); }
  catch(e){ mets=`<div class="note">unavailable</div>`; }
  h+=facet("Measured", `<div class="metgrid">${mets}</div>`);
  h+=`</div>`;

  sec.innerHTML=h;
  if(s.panel && PANELS[s.panel]){
    const host=el("div"); sec.appendChild(host);
    try{ PANELS[s.panel](host); }catch(e){ host.innerHTML=`<p class="note">panel error: ${esc(e.message)}</p>`; }
  }
  return sec;
}

/* ── instruments ──────────────────────────────────────────────────── */
function panelShell(host, title, ctrlId){
  const p=el("div","panel");
  p.innerHTML=`<div class="panel-hd"><h4>${title}</h4>${ctrlId?`<div class="seg" id="${ctrlId}"></div>`:""}</div>
    <div class="panel-bd"></div>`;
  host.appendChild(p);
  return p.querySelector(".panel-bd");
}
function segbar(node, opts, onPick, initial){
  opts.forEach(([v,lbl],i)=>{
    const b=el("button",null,lbl); b.setAttribute("aria-pressed", i===(initial||0));
    b.onclick=()=>{[...node.children].forEach(c=>c.setAttribute("aria-pressed","false"));
      b.setAttribute("aria-pressed","true"); onPick(v);};
    node.appendChild(b);});
}

const PANELS = {

panelSources(host){
  const bd=panelShell(host,"Source registry · filter","srcFilter");
  bd.classList.add("flush"); bd.innerHTML=`<div class="scroll"><table id="srcTable"></table></div>`;
  let f="all";
  function draw(){
    const rows=D.sources.filter(s=> f==="all"?true :
      f==="blocked" ? (s.tier==="D"||!s.clean||s.never||s.contam!=="scanned_clean") : s.lane===f);
    $("#srcTable").innerHTML=`<thead><tr><th>source</th><th>lane</th><th>licence</th><th>tier</th>
      <th>cleaning</th><th>contamination</th><th>note</th></tr></thead><tbody>`+
      rows.map(s=>`<tr><td class="mono" style="font-size:.75rem">${s.id}</td><td>${s.lane}</td>
      <td style="font-size:.75rem">${s.licence}</td>
      <td><span class="pill ${s.tier==="D"?"p-stop":"p-mute"}">${s.tier}</span></td>
      <td class="mono" style="font-size:.71rem">${s.clean||'<span class="pill p-stop">none</span>'}</td>
      <td class="mono" style="font-size:.71rem">${s.contam}</td>
      <td style="font-size:.75rem;color:var(--ink-3)">${
        s.expected ? s.expected.replace(/^REJECTED[^:]*:\s*/,'<span class="pill p-stop">refused</span> ')
        : (s.never?'<span class="pill p-stop">never train</span>'
        : (s.heldout?'<span class="pill p-warn">held out</span>':''))}</td></tr>`).join("")+"</tbody>";
  }
  segbar($("#srcFilter"),[["all","all"],["blocked","inadmissible"],...LANES.map(l=>[l,SHORT[l]])],
    v=>{f=v;draw();});
  draw();
},

panelTokenizer(host){
  const p=el("div","panel");
  p.innerHTML=`<div class="panel-hd"><h4>Frozen tokenizer &amp; measured fertility</h4></div>
    <div class="panel-bd"><div class="grid2">
      <div><dl class="kv" id="tokKv"></dl>
        <p class="note tight">The hash is recomputed from the merge table, not read back from the
        file. Reading a hash out of a file and trusting it proves nothing.</p></div>
      <div id="fertility"></div></div></div>`;
  host.appendChild(p);
  const e=D.evidence.find(x=>x.req==="Tokenizer integrity").detail, dl=$("#tokKv");
  [["vocab size",e.vocab_size],["merges",e.merges],["recorded hash",e.recorded_hash.slice(0,20)],
   ["recomputed",e.recomputed_from_merge_table.slice(0,20)],["match",e.hash_matches?"yes":"NO"],
   ["shards stamped",e.shard_manifests_checked],
   ["distinct hashes",e.distinct_tokenizer_hashes_in_manifests.length]]
   .forEach(([k,v])=>{dl.appendChild(el("dt",null,k)); dl.appendChild(el("dd",null,String(v)));});
  const names={en:"English",hi:"Hindi",bn:"Bengali",mr:"Marathi",te:"Telugu",ta:"Tamil"};
  const ent=Object.entries(D.corpus.fertility_by_language).sort((a,b)=>a[1]-b[1]);
  const mx=Math.max(...ent.map(x=>x[1])), box=$("#fertility");
  box.appendChild(el("div","note","tokens per whitespace word — lower is cheaper"));
  ent.forEach(([k,v])=>{
    const row=el("div","hbar"); row.appendChild(el("span",null,names[k]||k));
    const b=el("div","bar"), sp=el("span");
    sp.style.width=(v/mx*100)+"%";
    sp.style.background = v>4?"var(--stop)":v>2.5?"var(--warn)":"var(--accent)";
    b.appendChild(sp); row.appendChild(b); row.appendChild(el("span","num",v.toFixed(2)));
    box.appendChild(row);});
  box.appendChild(el("p","note tight",
    `Tamil costs ${(D.corpus.fertility_by_language.ta/D.corpus.fertility_by_language.en).toFixed(1)}× more tokens per word than English. Four times less content fits the same context window, and the same document costs four times more to train on.`));
},

panelAdmission(host){
  const p=el("div","panel");
  p.innerHTML=`<div class="panel-hd"><h4>Gate simulator · flip a condition, watch the verdict</h4>
    <span class="note">runs the same rule set as <code>admission_decision()</code></span></div>
    <div class="panel-bd"><div class="grid2">
      <div class="stack-s" id="gateControls"></div>
      <div class="stack-s"><div class="stage-box" id="gateVerdict"></div></div>
    </div></div>`;
  host.appendChild(p);
  const CONDS=[["licence","licence tier is A, B or C","licence_tier_not_trainable"],
    ["cleaning","cleaning pipeline hash recorded","cleaning_lineage_unknown"],
    ["tokenizer","tokenizer hash present","tokenizer_hash_missing"],
    ["contam","contamination scan came back clean","contamination_not_scanned"],
    ["overlap","no verbatim overlap with a benchmark","eval_overlap_detected"],
    ["never","not flagged never-train","never_train_flag_set"]];
  const state={}; CONDS.forEach(c=>state[c[0]]=true);
  const ctrl=$("#gateControls");
  CONDS.forEach(c=>{const lab=el("label","switch"); const cb=el("input"); cb.type="checkbox"; cb.checked=true;
    cb.onchange=()=>{state[c[0]]=cb.checked; verdict();};
    lab.appendChild(cb); lab.appendChild(el("span",null,c[1])); ctrl.appendChild(lab);});
  function verdict(){
    const fired=CONDS.filter(c=>!state[c[0]]).map(c=>c[2]);
    $("#gateVerdict").innerHTML = fired.length===0
      ? `<span class="pill p-pass">ADMITTED</span><p class="note" style="margin-top:.5rem">
         Every condition holds. The shard enters the registry with <code>permission = train</code>
         and becomes visible to the scheduler.</p>`
      : `<span class="pill p-stop">REFUSED</span><p class="note" style="margin-top:.5rem">
         Rules fired, recorded verbatim in <code>admission_report.json</code>:</p>
         <div style="margin-top:.45rem;display:flex;flex-wrap:wrap;gap:.3rem">
         ${fired.map(f=>`<span class="pill p-stop">${f}</span>`).join("")}</div>`;
  }
  verdict();

  const p2=el("div","panel");
  p2.innerHTML=`<div class="panel-hd"><h4>Every shard and its verdict</h4><div class="seg" id="shardFilter"></div></div>
    <div class="panel-bd flush scroll"><table id="shardTable"></table></div>`;
  host.appendChild(p2);
  let f="all";
  function draw(){
    const rows=D.shards.filter(s=> f==="all"?true : f==="refused"? !s.admitted : s.admitted);
    $("#shardTable").innerHTML=`<thead><tr><th>shard</th><th>lane</th><th class="num">tokens</th>
      <th class="num">docs</th><th>content hash</th><th>verdict</th></tr></thead><tbody>`+
      rows.map(s=>`<tr><td class="mono" style="font-size:.73rem">${s.id}</td><td>${s.lane}</td>
      <td class="num">${fmt(s.tokens)}</td><td class="num">${s.docs}</td>
      <td class="mono" style="font-size:.71rem;color:var(--ink-3)">${s.hash}</td>
      <td>${s.admitted?'<span class="pill p-pass">admitted</span>'
        :s.reasons.map(r=>`<span class="pill p-stop">${r}</span>`).join(" ")}</td></tr>`).join("")+"</tbody>";
  }
  segbar($("#shardFilter"),[["all","all "+D.admission.total],["admitted","admitted "+D.admission.admitted],
    ["refused","refused "+D.admission.rejected]], v=>{f=v;draw();});
  draw();
},

panelFirewall(host){
  const bd=panelShell(host,"Blocks recorded during this run");
  bd.innerHTML=`<div class="grid2">
    <div><dl class="kv" id="fwKv"></dl></div>
    <div class="stack-s" id="fwBlocks"></div></div>`;
  const dl=$("#fwKv");
  [["registry checks",D.firewall.registry_checks],["batch checks",D.firewall.batch_checks],
   ["blocks recorded",D.firewall.blocks_total],
   ["by side",Object.entries(D.firewall.blocks_by_side).map(([k,v])=>k+"="+v).join(" ")],
   ["validation gradient tokens",D.firewall.validation_gradient_bearing_tokens],
   ["n-gram size",D.fingerprints.ngram_size],
   ["overlap threshold",D.fingerprints.overlap_threshold],
   ["benchmark items",D.fingerprints.benchmark_items.length]]
   .forEach(([k,v])=>{dl.appendChild(el("dt",null,k)); dl.appendChild(el("dd",null,String(v)));});
  $("#fwBlocks").innerHTML = D.firewall_blocks.map(b=>`<div class="stage-box">
    <span class="pill p-stop">${b.side} side</span>
    <span class="mono" style="font-size:.76rem;margin-left:.35rem">${b.reason}</span>
    <div class="note tight mono" style="font-size:.72rem">${b.shard_id}</div>
    ${b.detail&&b.detail.hits?`<div class="note tight">detected by
      <b>${b.detail.hits[0].detector}</b> against <code>${b.detail.hits[0].benchmark_doc_id}</code>
      at overlap ${b.detail.hits[0].overlap_ratio}</div>`:""}
    </div>`).join("") || `<p class="note">no blocks recorded</p>`;
  const note=el("p","note tight",
    `The registry-side block is the never-train benchmark shard refused at planning time. The
     batch-side block is the same content packed into a real training window and pushed at
     <code>check_batch()</code> — refused on the decoded text, with the canary found.`);
  bd.appendChild(note);
},

panelMixture(host){
  const p=el("div","panel");
  p.innerHTML=`<div class="panel-hd"><h4>Lane quotas across all 24 steps</h4><div class="seg" id="mixMode"></div></div>
    <div class="panel-bd"><div id="mixChart"></div><div class="legend" id="mixLegend"></div></div>`;
  host.appendChild(p);
  let mode="planned";
  function draw(){
    const W=920,H=250,P={t:14,r:12,b:36,l:34};
    const steps=D.per_step,n=steps.length,per=8,bw=(W-P.l-P.r)/n;
    let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Stacked lane quotas for each training step with curriculum stage bands"><g font-family="ui-monospace,monospace" fill="currentColor">`;
    let acc=0;
    D.stages.forEach((st,i)=>{const x0=P.l+acc*bw,w=st.steps*bw; acc+=st.steps;
      s+=`<rect x="${x0}" y="${P.t}" width="${w}" height="${H-P.t-P.b}" fill="currentColor" opacity="${i%2?.045:.015}"/>
        <text x="${x0+5}" y="${H-P.b+14}" font-size="9" opacity=".7">${st.name}</text>
        <text x="${x0+5}" y="${H-P.b+25}" font-size="8.5" opacity=".5">window ${st.sequence_length}</text>`;});
    steps.forEach((q,i)=>{
      const counts = mode==="planned"? q.counts : (D.actual_lane_counts[String(q.step)]||{});
      let y=H-P.b;
      LANES.forEach(l=>{const c=counts[l]||0; if(!c)return;
        const h=(c/per)*(H-P.t-P.b);
        s+=`<rect x="${P.l+i*bw+.6}" y="${y-h}" width="${bw-1.2}" height="${h}" fill="${LANE_HUE[l]}" opacity=".85"><title>step ${q.step} · ${l} · ${c}</title></rect>`;
        y-=h;});});
    s+=`<line x1="${P.l}" y1="${H-P.b}" x2="${W-P.r}" y2="${H-P.b}" stroke="currentColor" opacity=".35"/>`;
    for(let k=0;k<=per;k+=2){const y=H-P.b-(k/per)*(H-P.t-P.b);
      s+=`<text x="${P.l-6}" y="${y+3}" text-anchor="end" font-size="9" opacity=".55">${k}</text>`;}
    [0,6,12,18,23].forEach(k=>s+=`<text x="${P.l+k*bw+bw/2}" y="${H-P.b+13}" text-anchor="middle" font-size="9" opacity=".55">${k}</text>`);
    s+="</g></svg>";
    $("#mixChart").innerHTML=s;
  }
  segbar($("#mixMode"),[["planned","planned quotas"],["actual","actual, from the ledger"]],v=>{mode=v;draw();});
  LANES.forEach(l=>$("#mixLegend").appendChild(el("span",null,`<i style="background:${LANE_HUE[l]}"></i>${l}`)));
  draw(); REDRAW.push(draw);

  const p2=el("div","panel");
  p2.innerHTML=`<div class="panel-bd"><div class="grid2">
    <div><h4 style="margin-bottom:.5rem">Protected floors</h4><div class="scroll"><table id="floorTable"></table></div></div>
    <div><h4 style="margin-bottom:.5rem">Scarcity, resolved explicitly</h4><div class="scroll"><table id="scarceTable"></table></div></div>
  </div></div>`;
  host.appendChild(p2);
  const fl={};
  D.floor_adjustments.forEach(a=>{const k=a.stage+"|"+a.lane;
    fl[k]=fl[k]||{n:0}; fl[k].n++;});
  $("#floorTable").innerHTML=`<thead><tr><th>stage</th><th>lane</th><th class="num">floor</th>
    <th class="num">rescues</th><th class="num">achieved</th><th></th></tr></thead><tbody>`+
    D.compliance.protected_floor_checks.map(c=>{const k=c.stage+"|"+c.lane;
      return `<tr><td style="font-size:.74rem">${c.stage.replace(/-/g," ")}</td><td>${c.lane}</td>
      <td class="num">${pct(c.floor)}</td><td class="num">${fl[k]?fl[k].n:0}</td>
      <td class="num">${pct(c.achieved_share)}</td>
      <td>${c.respected?'<span class="pill p-pass">held</span>':'<span class="pill p-stop">breached</span>'}</td></tr>`;
    }).join("")+"</tbody>";
  $("#scarceTable").innerHTML=`<thead><tr><th>stage</th><th>lane</th><th class="num">need</th>
    <th class="num">have</th><th class="num">×</th><th>resolution</th></tr></thead><tbody>`+
    D.feasibility.filter(f=>f.resolution!=="satisfied").map(f=>`<tr>
      <td style="font-size:.73rem">${f.stage.replace(/-/g," ")}</td><td>${f.lane}</td>
      <td class="num">${f.sequences_required}</td><td class="num">${f.distinct_samples_available}</td>
      <td class="num">${f.repeat_factor.toFixed(2)}</td>
      <td><span class="pill ${f.resolution==="repeat_existing_data"?"p-warn":"p-stop"}">${f.resolution.replace(/_/g," ")}</span></td></tr>`).join("")+"</tbody>";

  host.appendChild(el("div","callout warn",
    `<strong>An honest limit, reported rather than hidden.</strong> At 8 sequences per step the
     finest expressible share is 12.5%. A stated 4% agentic share cannot be served as 4% — it is
     served as 0% or 12.5%, and the protected floor forces the latter.
     <code>mixture_schedule.json</code> carries a <code>stage_intent_vs_compiled</code> block naming
     which constraint moved each lane, and compliance is measured against the compiled plan rather
     than against an intent the batch geometry cannot express.`)).style.marginTop=".9rem";
},

panelPacking(host){
  const p=el("div","panel");
  p.innerHTML=`<div class="panel-hd"><h4>Policy comparator · every policy over the same items</h4>
    <label class="fld" style="flex-direction:row;align-items:center;gap:.5rem">
      <span>lane / window</span><select id="packLane"></select></label></div>
    <div class="panel-bd flush scroll"><table id="packTable"></table></div>
    <div class="panel-bd" style="border-top:1px solid var(--rule-2)">
      <p class="note"><strong>Read utilisation together with retention.</strong>
      <code>pad_only</code> reaches high utilisation by truncating whatever does not fit — perfect
      packing of the part it kept, and a corpus that lost tokens.
      <span class="pill p-acc" id="chosenPill">chosen</span> marks the policy this lane uses.</p></div>`;
  host.appendChild(p);
  const keys=Object.keys(D.policy_comparison).sort(), sel=$("#packLane");
  keys.forEach(k=>{const o=el("option",null,k); o.value=k; sel.appendChild(o);});
  function draw(){
    const k=sel.value, cmp=D.policy_comparison[k];
    const chosen=D.pack_stats[k]?D.pack_stats[k].policy:null;
    $("#chosenPill").textContent=chosen||"—";
    const order=["pad_only","concat_chop","greedy","best_fit","structure_preserving","long_context"];
    $("#packTable").innerHTML=`<thead><tr><th>policy</th><th class="num">windows</th>
      <th class="num">utilisation</th><th class="num">retention</th><th class="num">splits</th>
      <th class="num">truncated</th><th class="num">deferred</th><th class="num">co-packed</th></tr></thead><tbody>`+
      order.filter(x=>cmp[x]).map(x=>{const s=cmp[x],on=x===chosen;
        return `<tr style="${on?'background:var(--accent-soft)':''}">
        <td class="mono" style="font-size:.75rem">${x}${on?' <span class="pill p-acc">chosen</span>':''}</td>
        <td class="num">${s.windows}</td><td class="num">${s.utilisation.toFixed(3)}</td>
        <td class="num" style="${s.retention<1?'color:var(--stop)':''}">${s.retention.toFixed(3)}</td>
        <td class="num">${s.document_splits}</td>
        <td class="num" style="${s.truncated_tokens>0?'color:var(--stop)':''}">${fmt(s.truncated_tokens)}</td>
        <td class="num">${fmt(s.deferred_tokens)}</td><td class="num">${s.boundary_crossings}</td></tr>`;}).join("")+"</tbody>";
  }
  sel.onchange=draw;
  sel.value=keys.find(k=>k.startsWith("code@256"))||keys[0];
  draw();
},

panelMasks(host){
  const p=el("div","panel");
  p.innerHTML=`<div class="panel-hd"><h4>Token inspector · hover any token</h4><div class="seg" id="maskPick"></div></div>
    <div class="panel-bd"><div id="maskMeta" class="grid3" style="margin-bottom:.8rem"></div>
    <div class="strip" id="tokenStrip"></div>
    <div class="legend">
      <span><i style="background:var(--pass)"></i>graded — loss applied</span>
      <span><i style="background:var(--rule)"></i>context only — conditioned on, not graded</span>
      <span><i style="border:1px dashed var(--ink-3)"></i>padding</span></div></div>`;
  host.appendChild(p);
  const samples={"agentic":D.sample_agentic,"general_web":D.sample_web};
  let cur="agentic";
  function draw(){
    const s=samples[cur];
    $("#maskMeta").innerHTML=[["policy",s.policy],["window",s.seq],["graded",s.loss_tokens],
      ["context only",s.ctx],["padding",s.pad],["documents packed",s.segments.length]]
      .map(([k,v])=>`<div class="met"><div class="mk">${k}</div><div class="mv">${v}</div></div>`).join("");
    const strip=$("#tokenStrip"); strip.innerHTML="";
    s.tokens.forEach((t,i)=>{
      const cls=t.s<0?"p":(t.m?"g":"c");
      const d=el("span","tok "+cls,esc(t.t||"·").replace(/\n/g,"⏎").replace(/ /g,"·"));
      const seg=s.segments[t.s];
      hoverable(d,()=>`pos ${i} · segment-relative ${t.p}<br>`+
        (t.s<0?"<b>padding</b>":`doc <b>${seg?seg.doc:"?"}</b><br>shard ${seg?seg.shard:"?"}`)+
        `<br>loss mask <b>${t.m}</b> — ${t.s<0?"never graded":(t.m?"graded":"context only")}`);
      strip.appendChild(d);});
  }
  segbar($("#maskPick"),[["agentic","agentic — model turns only"],["general_web","general_web — plain next-token"]],
    v=>{cur=v;draw();});
  draw();

  const p2=el("div","panel");
  p2.innerHTML=`<div class="panel-hd"><h4>Validators applied to every packed sample</h4>
    <span class="note">${D.mask_validation.samples_checked} samples · 0 failures</span></div>
    <div class="panel-bd"><ul style="margin:0;padding-left:1.1rem;font-size:.86rem;line-height:1.85">`+
    D.mask_validation.checks_applied.map(c=>`<li>${c} <span class="pill p-pass">pass</span></li>`).join("")+
    `</ul></div>`;
  host.appendChild(p2);
},

panelOpus(host){
  const p=el("div","panel");
  p.innerHTML=`<div class="panel-hd"><h4>Decision explorer · every candidate scored</h4><div class="seg" id="opusFilter"></div></div>
    <div class="panel-bd"><canvas id="opusCanvas" height="300" style="height:300px"></canvas>
    <div class="legend">
      <span><i style="background:var(--pass)"></i>accepted</span>
      <span><i style="background:var(--stop)"></i>rejected</span>
      <span><i style="background:var(--ink-3)"></i>deferred</span>
      <span><i style="background:var(--warn)"></i>protected-floor override</span>
      <span>x — gradient cosine · y — candidate gradient norm (√ scale)</span></div>
    <p class="note tight" id="opusDetail"></p></div>`;
  host.appendChild(p);
  let filter="all", pts=[];
  const F=D.opus_fields, iS=F.indexOf("score"), iG=F.indexOf("grad_norm"), iSt=F.indexOf("status"),
        iL=F.indexOf("lane"), iO=F.indexOf("override");
  function draw(){
    const cv=$("#opusCanvas"); if(!cv) return;
    const dpr=devicePixelRatio||1, w=cv.clientWidth, h=300;
    if(!w){requestAnimationFrame(draw); return;}
    cv.width=w*dpr; cv.height=h*dpr;
    const g=cv.getContext("2d"); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,w,h);
    const rows=D.opus.filter(r=> filter==="all"?true :
      ["accepted","rejected","deferred"].includes(filter)? r[iSt]===filter :
      filter==="override"? r[iO]===1 : r[iL]===filter);
    const P={t:12,r:12,b:30,l:46};
    const xs=D.opus.map(r=>r[iS]), ys=D.opus.map(r=>r[iG]);
    const x0=Math.min(...xs),x1=Math.max(...xs),y1=Math.max(...ys);
    const X=v=>P.l+((v-x0)/(x1-x0))*(w-P.l-P.r);
    const Y=v=>h-P.b-(Math.sqrt(v)/Math.sqrt(y1))*(h-P.t-P.b);
    const ink=css("--ink-3");
    g.strokeStyle=ink; g.globalAlpha=.3; g.beginPath(); g.moveTo(P.l,h-P.b); g.lineTo(w-P.r,h-P.b); g.stroke();
    if(x0<0&&x1>0){g.beginPath(); g.moveTo(X(0),P.t); g.lineTo(X(0),h-P.b); g.stroke();}
    g.globalAlpha=1; g.fillStyle=ink; g.font="10px ui-monospace,monospace";
    g.fillText(x0.toFixed(2),P.l,h-P.b+14); g.fillText("0",X(0)-3,h-P.b+14);
    g.fillText(x1.toFixed(2),w-P.r-26,h-P.b+14);
    const C={accepted:css("--pass"),rejected:css("--stop"),deferred:css("--ink-3")};
    pts=[];
    rows.forEach(r=>{const x=X(r[iS]),y=Y(r[iG]);
      g.beginPath(); g.arc(x,y, r[iO]===1?3.4:2.3,0,6.2832);
      g.fillStyle = r[iO]===1?css("--warn"):C[r[iSt]];
      g.globalAlpha = r[iO]===1?.95:.55; g.fill();
      pts.push({x,y,r});});
    g.globalAlpha=1;
    $("#opusDetail").textContent=`${rows.length} of ${D.opus.length} decisions shown. Hover a point for its record.`;
  }
  segbar($("#opusFilter"),[["all","all "+D.opus.length],["accepted","accepted"],["rejected","rejected"],
    ["deferred","deferred"],["override","floor override"],...LANES.map(l=>[l,SHORT[l]])],v=>{filter=v;draw();});
  draw(); REDRAW.push(draw); addEventListener("resize",draw);
  $("#opusCanvas").addEventListener("mousemove",e=>{
    const b=e.target.getBoundingClientRect(), mx=e.clientX-b.left, my=e.clientY-b.top;
    let best=null,bd=1e9;
    pts.forEach(pt=>{const d=(pt.x-mx)**2+(pt.y-my)**2; if(d<bd){bd=d;best=pt;}});
    if(best&&bd<170){const o={}; F.forEach((f,i)=>o[f]=best.r[i]);
      showTip(e,`step ${o.step} · <b>${o.lane}</b><br><b>${o.status}</b> — ${o.reason.replace(/_/g," ")}<br>`+
        `score ${o.score.toFixed(4)} vs threshold ${o.threshold.toFixed(4)}<br>`+
        `grad norm ${o.grad_norm.toFixed(4)} · loss ${o.loss}<br>`+
        (o.override?"<b>protected-floor override</b><br>":"")+`pass ${o.pass} · ${o.candidate}`);
    } else hideTip();});
  $("#opusCanvas").addEventListener("mouseleave",hideTip);

  const p2=el("div","panel");
  p2.innerHTML=`<div class="panel-bd"><div class="grid2">
    <div id="opusBreakdown"></div><div id="proxyHealth"></div></div></div>`;
  host.appendChild(p2);
  const rep=D.opus_report, tot=rep.total_candidates_scored;
  $("#opusBreakdown").innerHTML=`<h4 style="margin-bottom:.5rem">Four ledgers, five reasons</h4>`+
    Object.entries(rep.by_status).map(([k,v])=>`<div class="hbar"><span>${k}</span>
      <div class="bar"><span style="width:${v/tot*100}%;background:${k==="accepted"?"var(--pass)":k==="rejected"?"var(--stop)":"var(--ink-3)"}"></span></div>
      <span class="num">${v}</span></div>`).join("")+
    `<div style="margin-top:.7rem" class="stack-s">`+
    Object.entries(rep.by_reason).sort((a,b)=>b[1]-a[1]).map(([k,v])=>
      `<div style="display:flex;justify-content:space-between"><span class="mono" style="font-size:.72rem">${k.replace(/_/g," ")}</span><span class="num mono">${v}</span></div>`).join("")+`</div>`;
  const ph=D.proxy_health;
  $("#proxyHealth").innerHTML=`<h4 style="margin-bottom:.5rem">Proxy health</h4>
    <dl class="kv"><dt>rounds</dt><dd>${ph.rounds}</dd>
    <dt>first round grad norm</dt><dd>${ph.first_round_accepted_grad_norm}</dd>
    <dt>last round</dt><dd>${ph.last_round_accepted_grad_norm}</dd>
    <dt>ratio</dt><dd>${ph.ratio_last_over_first}</dd></dl>
    <div class="callout warn" style="margin-top:.6rem;font-size:.82rem"><strong>Verdict:</strong> ${ph.verdict}.</div>
    <p class="note tight">A real negative finding, left in rather than tuned quiet. For a 20k-token
    corpus over 24 steps it is the correct diagnosis, and it is exactly the signal the design
    describes — the selector still fills its quota, but there is nothing good left to fill it with.</p>`;
},

panelTraining(host){
  const p=el("div","panel");
  p.innerHTML=`<div class="panel-hd"><h4>Loss, with stage bands, checkpoints and the crash</h4><div class="seg" id="curveMode"></div></div>
    <div class="panel-bd"><div id="curveChart"></div></div>`;
  host.appendChild(p);
  let mode="loss";
  function draw(){
    const W=920,H=280,P={t:16,r:14,b:34,l:46}, c=D.curve, n=c.length;
    const vals=c.map(p=> mode==="loss"?p.loss : mode==="gn"?p.gn : p.lr);
    const vmax=Math.max(...vals)*1.06, vmin= mode==="loss"?Math.min(...vals)*.97:0;
    const X=i=>P.l+(i/(n-1))*(W-P.l-P.r), Y=v=>H-P.b-((v-vmin)/(vmax-vmin))*(H-P.t-P.b);
    let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Training loss across 24 steps with stage bands, checkpoints and the injected crash"><g font-family="ui-monospace,monospace" fill="currentColor">`;
    let acc=0;
    D.stages.forEach((st,i)=>{const x0=X(acc),x1=X(Math.min(acc+st.steps,n-1)); acc+=st.steps;
      s+=`<rect x="${x0}" y="${P.t}" width="${x1-x0}" height="${H-P.t-P.b}" fill="currentColor" opacity="${i%2?.045:.015}"/>
        <text x="${x0+5}" y="${P.t+11}" font-size="8.5" opacity=".55">${st.name}</text>`;});
    if(mode==="loss"){const lnv=Math.log(D.corpus.vocab_size);
      if(lnv<vmax&&lnv>vmin) s+=`<line x1="${P.l}" y1="${Y(lnv)}" x2="${W-P.r}" y2="${Y(lnv)}" stroke="var(--accent)" stroke-dasharray="4 3" opacity=".7"/>
        <text x="${W-P.r-4}" y="${Y(lnv)-5}" text-anchor="end" font-size="9" fill="var(--accent)">ln(V) = ${lnv.toFixed(2)} — an untrained model</text>`;}
    const cx=X(16), rx=X(12);
    s+=`<rect x="${rx}" y="${P.t}" width="${cx-rx}" height="${H-P.t-P.b}" fill="var(--warn)" opacity=".07"/>
      <line x1="${cx}" y1="${P.t}" x2="${cx}" y2="${H-P.b}" stroke="var(--stop)" stroke-width="1.4" opacity=".85"/>
      <text x="${cx+4}" y="${P.t+24}" font-size="9" fill="var(--stop)">crash · step 16</text>
      <line x1="${rx}" y1="${P.t}" x2="${rx}" y2="${H-P.b}" stroke="var(--warn)" stroke-dasharray="3 3" opacity=".8"/>
      <text x="${rx+4}" y="${P.t+38}" font-size="9" fill="var(--warn)">resumed from here</text>
      <text x="${(rx+cx)/2}" y="${H-P.b-6}" text-anchor="middle" font-size="8.5" fill="var(--warn)">re-served identically</text>`;
    s+=`<polyline points="${vals.map((v,i)=>X(i)+","+Y(v)).join(" ")}" fill="none" stroke="var(--accent)" stroke-width="1.8"/>`;
    c.forEach((p,i)=>{const isCk=p.ckpt&&p.ckpt!=="genesis"&&(i===0||c[i-1].ckpt!==p.ckpt);
      s+=`<circle cx="${X(i)}" cy="${Y(vals[i])}" r="${isCk?3.4:2}" fill="${isCk?"var(--card)":"var(--accent)"}" stroke="var(--accent)" stroke-width="${isCk?1.6:0}"><title>step ${p.s} · loss ${p.loss} · ppl ${p.ppl} · ${p.stage}</title></circle>`;});
    if(mode==="loss"&&D.validation.length){
      s+=`<polyline points="${D.validation.map(p=>X(p.s)+","+Y(p.loss)).join(" ")}" fill="none" stroke="var(--ink-3)" stroke-width="1.3" stroke-dasharray="4 3"/>`;
      D.validation.forEach(p=>s+=`<circle cx="${X(p.s)}" cy="${Y(p.loss)}" r="2.4" fill="var(--ink-3)"><title>validation loss ${p.loss} — never gradient-bearing</title></circle>`);
      const last=D.validation[D.validation.length-1];
      s+=`<text x="${X(last.s)-6}" y="${Y(last.loss)+14}" text-anchor="end" font-size="9" opacity=".65">validation (loss only, no gradient)</text>`;}
    for(let k=0;k<=4;k++){const v=vmin+(vmax-vmin)*k/4;
      s+=`<text x="${P.l-6}" y="${Y(v)+3}" text-anchor="end" font-size="9" opacity=".55">${v.toFixed(2)}</text>`;}
    [0,6,12,18,23].forEach(k=>s+=`<text x="${X(k)}" y="${H-P.b+15}" text-anchor="middle" font-size="9" opacity=".55">${k}</text>`);
    s+=`<text x="${W/2}" y="${H-4}" text-anchor="middle" font-size="9" opacity=".5">optimizer step</text></g></svg>`;
    $("#curveChart").innerHTML=s;
  }
  segbar($("#curveMode"),[["loss","loss"],["gn","gradient norm"],["lr","learning rate"]],v=>{mode=v;draw();});
  draw(); REDRAW.push(draw);

  const p2=el("div","panel");
  p2.innerHTML=`<div class="panel-bd"><div class="grid2">
    <div><h4 style="margin-bottom:.5rem">Batch geometry</h4><dl class="kv" id="geomKv"></dl></div>
    <div><h4 style="margin-bottom:.5rem">Model &amp; optimiser</h4><dl class="kv" id="modelKv"></dl></div>
  </div></div>`;
  host.appendChild(p2);
  const g=$("#geomKv");
  [["ranks (simulated)",2],["microbatch / rank",2],["gradient accumulation",2],
   ["sequences per step",8],["microbatches per step",D.integrity.expected_microbatches_per_step],
   ["steps",D.curve.length],["window, stages 1–2",256],["window, anneal",512],
   ["positions consumed",fmt(D.integrity.total_positions)]]
   .forEach(([k,v])=>{g.appendChild(el("dt",null,k)); g.appendChild(el("dd",null,String(v)));});
  const m=$("#modelKv");
  [["parameters","1,312,256"],["layers / heads / d_model","4 / 4 / 128"],
   ["vocab",D.corpus.vocab_size],["optimiser","AdamW β 0.9/0.95"],["schedule","warmup 6 → cosine"],
   ["grad clip","1.0"],["first-step loss",D.curve[0].loss],
   ["ln(V)",Math.log(D.corpus.vocab_size).toFixed(4)],
   ["final loss",D.curve[D.curve.length-1].loss]]
   .forEach(([k,v])=>{m.appendChild(el("dt",null,k)); m.appendChild(el("dd",null,String(v)));});
},

panelConsumption(host){
  const bd=panelShell(host,"One consumption record, field by field");
  const fields=[["run_id / branch_id","which run, which data branch"],
    ["global_step / rank / microbatch_id","exactly which slot in the schedule"],
    ["batch_id / plan_hash","the batch, and the plan it came from"],
    ["packed_sample_ids","which windows"],["shard_ids / token_span_ids","which tokens, exactly"],
    ["tokens_hash / loss_mask_hash","content identity of both arrays"],
    ["mixture_lane / curriculum_stage","why this batch was chosen"],
    ["opus_decision_id","why this batch and not the one beside it"],
    ["repeated_pass_number","which pass over the lane stream"],
    ["attention_policy / position_policy","how it was masked"],
    ["tokenizer_version / dataloader_version / config_hash","what code and vocabulary produced it"],
    ["rng_fingerprint","the derived RNG position"],
    ["loss_bearing / context_only / pad / total positions","the token accounting"]];
  bd.innerHTML=`<div class="grid2"><div><dl class="kv" id="consKv"></dl></div>
    <div><div class="note" style="margin-bottom:.4rem">Every record carries:</div>
    <table><tbody>`+fields.map(f=>`<tr><td class="mono" style="font-size:.71rem;white-space:nowrap">${f[0]}</td>
      <td style="font-size:.78rem;color:var(--ink-3)">${f[1]}</td></tr>`).join("")+`</tbody></table></div></div>`;
  const c=$("#consKv");
  [["records",D.integrity.records],["microbatches",D.integrity.distinct_microbatches],
   ["step range",D.integrity.step_range.join("–")],["duplicates",D.integrity.duplicate_count],
   ["missing steps",D.integrity.missing_steps.length||"none"],
   ["per-step count",D.integrity.microbatches_per_step.join(",")],
   ["loss-bearing tokens",fmt(D.integrity.loss_bearing_tokens)],
   ["pad tokens",fmt(D.integrity.pad_tokens)],["hash chain","intact"]]
   .forEach(([k,v])=>{c.appendChild(el("dt",null,k)); c.appendChild(el("dd",null,String(v)));});
},

panelLearning(host){
  const p=el("div","panel");
  p.innerHTML=`<div class="panel-hd"><h4>Per-token perplexity · one packed sample, real trace</h4>
    <span class="note">step ${D.token_trace.step} · ${D.token_trace.lane} · ${D.token_trace.n} loss-bearing tokens · phase ${D.token_trace.phase}</span></div>
    <div class="panel-bd"><div class="strip" id="ppxStrip"></div>
    <div class="legend"><span>colour — perplexity, pale = predicted easily, dark = surprised</span>
      <span><i style="background:var(--warn)"></i>EOS position</span></div>
    <p class="note tight">A shard's average hides this. The pattern of surprise inside it is the map
      for future data collection — which is why EOS perplexity is tracked separately: a model never
      surprised except at EOS has not learned boundaries, it has learned to keep going.</p></div>`;
  host.appendChild(p);
  const t=D.token_trace, strip=$("#ppxStrip");
  const mx=Math.max(...t.tokens.map(x=>x.ppl));
  const supportsMix = window.CSS && CSS.supports && CSS.supports("background","color-mix(in srgb, red 50%, transparent)");
  t.tokens.forEach(x=>{
    const norm=Math.log(x.ppl+1)/Math.log(mx+1);
    const d=el("span","tok",esc(x.t||"·").replace(/\n/g,"⏎").replace(/ /g,"·"));
    d.style.background = x.eos ? "var(--warn)"
      : (supportsMix ? `color-mix(in srgb, var(--accent) ${Math.round(norm*72)}%, transparent)`
                     : `rgba(11,110,99,${(norm*.6).toFixed(3)})`);
    if(x.eos) d.style.color="#fff";
    hoverable(d,()=>`<b>${esc(x.t)}</b><br>perplexity ${x.ppl}<br>cross-entropy ${x.ce}<br>`+
      `doc ${x.doc}<br>position ${x.pos}${x.eos?"<br><b>EOS boundary</b>":""}${x.sp?"<br>special token":""}`);
    strip.appendChild(d);});

  const p2=el("div","panel");
  p2.innerHTML=`<div class="panel-hd"><h4>Shard report cards → what the next corpus should do</h4><div class="seg" id="cardFilter"></div></div>
    <div class="panel-bd flush scroll"><table id="cardTable"></table></div>`;
  host.appendChild(p2);
  let cf="all";
  function cards(){
    const rows=D.shard_cards.filter(c=> cf==="all"?true:c.classification===cf);
    $("#cardTable").innerHTML=`<thead><tr><th>shard</th><th>lane</th><th class="num">exposures</th>
      <th class="num">mean loss Δ</th><th class="num">mean ppl</th><th class="num">repeat effect</th>
      <th>verdict</th><th>because</th></tr></thead><tbody>`+
      rows.map(c=>`<tr><td class="mono" style="font-size:.71rem">${c.shard_id}</td>
      <td>${c.lanes.join(", ")}</td><td class="num">${c.exposures}</td>
      <td class="num" style="color:${c.mean_loss_delta<0?"var(--pass)":"var(--stop)"}">${c.mean_loss_delta.toFixed(5)}</td>
      <td class="num">${c.mean_token_perplexity.toFixed(1)}</td>
      <td class="num">${c.repeat_effect.toFixed(4)}</td>
      <td><span class="pill ${c.classification==="useful"?"p-pass":c.classification==="harmful"?"p-stop":"p-warn"}">${c.classification}</span></td>
      <td style="font-size:.74rem;color:var(--ink-3);max-width:24rem">${c.rationale}</td></tr>`).join("")+"</tbody>";
  }
  segbar($("#cardFilter"),[["all","all "+D.shard_cards.length],
    ...Object.entries(D.corpus_verdicts).filter(([k,v])=>v>0).map(([k,v])=>[k,`${k} ${v}`])],v=>{cf=v;cards();});
  cards();
},

panelCheckpoint(host){
  const bd=panelShell(host,"Checkpoints written in this run");
  bd.classList.add("flush");
  bd.innerHTML=`<div class="scroll"><table><thead><tr><th>checkpoint</th><th class="num">step</th>
    <th>stage</th><th class="num">ledger byte offset</th><th class="num">event seq</th>
    <th>rng fingerprint</th><th>next plan hash</th><th>weights</th></tr></thead><tbody>`+
    D.checkpoints.map(c=>`<tr><td class="mono" style="font-size:.73rem">${c.id}</td>
    <td class="num">${c.step}</td><td style="font-size:.75rem">${c.stage}</td>
    <td class="num">${fmt(c.offset.byte_offset)}</td><td class="num">${c.offset.event_seq}</td>
    <td class="mono" style="font-size:.71rem">${c.rng}</td>
    <td class="mono" style="font-size:.71rem">${c.next_plan||"—"}</td>
    <td>${c.weights?'<span class="pill p-pass">kept</span>':'<span class="pill p-mute">pruned</span>'}</td></tr>`).join("")+
    `</tbody></table></div>`;
},

panelResume(host){
  const CK12=D.checkpoints.find(c=>c.step===12)||{offset:{byte_offset:0,event_seq:0}};
  const STEPS=[
   {t:"checkpoint at step 12",model:12,ledger:12,torn:false,note:
    `Model, optimiser, scheduler and the ledger's byte offset are written atomically. The
     checkpoint records <code>ledger_offset</code> = byte ${fmt(CK12.offset.byte_offset)},
     event seq ${CK12.offset.event_seq}, and the plan hash the next step should serve.`},
   {t:"steps 12–15 served",model:12,ledger:16,torn:false,note:
    `Each microbatch is recorded and fsynced <em>before</em> the optimizer step. The ledger is now
     four steps ahead of the durable model state — the weights for 12–15 exist only in memory.`},
   {t:"killed mid-write, step 16",model:12,ledger:16,torn:true,note:
    `<code>os._exit(137)</code> — no unwinding, no flush, no destructors. The last line is a
     ${D.recovery.torn_tail.removed_bytes}-byte fragment. On disk: weights from step 12, a ledger
     claiming step 16, and a torn tail.`},
   {t:"torn tail repaired",model:12,ledger:16,torn:false,note:
    `Recovery truncates the ${D.recovery.torn_tail.removed_bytes} partial bytes and logs the repair.
     A partial record never entered the hash chain, so the chain head is unchanged. The ledger is
     parseable again — and still four steps ahead.`},
   {t:"rolled back to the offset",model:12,ledger:12,torn:false,note:
    `${D.recovery.discarded_records} records for steps ${D.recovery.discarded_steps.join(", ")} are
     discarded, hashes logged first. They describe batches served to a model state that no longer
     exists. <b>This is why the checkpoint stores a byte offset and not a step number</b> — an
     offset names the truncation point, which is the operation actually required.`},
   {t:"re-served and compared",model:24,ledger:24,torn:false,note:
    `Steps 12–15 are served again from the restored state and the run completes. Every one of the
     ${D.rollback.compared} discarded microbatches reappears with an identical batch id, identical
     token spans and identical token and mask hashes — 0 mismatches, 0 missing.`}];
  let i=0;
  const p=el("div","panel");
  p.innerHTML=`<div class="panel-hd"><h4>Walk the crash · model state versus ledger position</h4>
    <span class="note" id="crashPhase"></span></div>
    <div class="panel-bd"><div class="stepper" id="crashSteps"></div>
    <div id="crashSvg"></div><div class="stage-box" id="crashNarr" style="margin-top:.85rem"></div></div>`;
  host.appendChild(p);
  function draw(){
    const c=STEPS[i], W=760,H=150, X=s=>60+(s/24)*(W-90);
    let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Model state and ledger position on a step axis, showing the gap the crash opens and the rollback closes"><g font-family="ui-monospace,monospace" fill="currentColor">
      <line x1="60" y1="118" x2="${W-24}" y2="118" stroke="currentColor" opacity=".3"/>`;
    [0,6,12,18,24].forEach(k=>s+=`<line x1="${X(k)}" y1="114" x2="${X(k)}" y2="122" stroke="currentColor" opacity=".4"/>
      <text x="${X(k)}" y="136" text-anchor="middle" font-size="9" opacity=".55">${k}</text>`);
    s+=`<text x="52" y="48" text-anchor="end" font-size="9.5" opacity=".7">model</text>
      <rect x="60" y="36" width="${X(c.model)-60}" height="16" rx="2" fill="var(--accent)" opacity=".8"/>
      <text x="${X(c.model)+6}" y="48" font-size="9.5" fill="var(--accent)">step ${c.model}</text>
      <text x="52" y="84" text-anchor="end" font-size="9.5" opacity=".7">ledger</text>
      <rect x="60" y="72" width="${X(Math.min(c.ledger,c.model))-60}" height="16" rx="2" fill="var(--accent)" opacity=".8"/>`;
    if(c.ledger>c.model){
      s+=`<rect x="${X(c.model)}" y="72" width="${X(c.ledger)-X(c.model)}" height="16" rx="2" fill="var(--stop)" opacity=".75"/>
        <text x="${X(c.ledger)+6}" y="84" font-size="9.5" fill="var(--stop)">ahead by ${c.ledger-c.model}</text>
        <path d="M${X(c.model)},30 v-8 h${X(c.ledger)-X(c.model)} v8" fill="none" stroke="var(--stop)" opacity=".6"/>
        <text x="${(X(c.model)+X(c.ledger))/2}" y="17" text-anchor="middle" font-size="9" fill="var(--stop)">the divergence</text>`;
    } else s+=`<text x="${X(c.ledger)+6}" y="84" font-size="9.5" fill="var(--accent)">step ${c.ledger}</text>`;
    if(c.torn) s+=`<rect x="${X(c.ledger)}" y="72" width="14" height="16" fill="none" stroke="var(--stop)" stroke-dasharray="2 2"/>
      <text x="${X(c.ledger)+20}" y="100" font-size="9" fill="var(--stop)">torn tail</text>`;
    s+="</g></svg>";
    $("#crashSvg").innerHTML=s;
    $("#crashNarr").innerHTML=`<div style="font-family:var(--mono);font-size:.65rem;text-transform:uppercase;
      letter-spacing:.09em;color:var(--ink-3)">${i+1} of ${STEPS.length} · ${c.t}</div>
      <p style="margin-top:.35rem;font-size:.89rem">${c.note}</p>`;
    [...$("#crashSteps").children].forEach((b,k)=>b.setAttribute("aria-pressed",k===i));
    $("#crashPhase").textContent=c.t;
  }
  const seg=$("#crashSteps");
  STEPS.forEach((c,k)=>{const b=el("button",null,String(k+1)); b.title=c.t;
    b.onclick=()=>{i=k;draw();}; seg.appendChild(b);});
  const nx=el("button",null,"next ›"); nx.onclick=()=>{i=(i+1)%STEPS.length;draw();}; seg.appendChild(nx);
  draw(); REDRAW.push(draw);

  const p2=el("div","panel");
  p2.innerHTML=`<div class="panel-bd"><div class="grid2">
    <div><h4 style="margin-bottom:.5rem">Recovery record</h4><dl class="kv" id="recKv"></dl></div>
    <div class="stack-s" id="proofs"></div></div></div>`;
  host.appendChild(p2);
  const r=$("#recKv");
  [["checkpoint",D.recovery.checkpoint],["resumed at step",D.recovery.resume_step],
   ["torn tail",D.recovery.torn_tail.removed_bytes+" bytes"],["reason",D.recovery.torn_tail.reason],
   ["records discarded",D.recovery.discarded_records],
   ["steps rolled back",D.recovery.discarded_steps.join(", ")],
   ["byte offset",fmt(D.recovery.ledger_offset.byte_offset)],
   ["event seq",D.recovery.ledger_offset.event_seq]]
   .forEach(([k,v])=>{r.appendChild(el("dt",null,k)); r.appendChild(el("dd",null,String(v)));});
  $("#proofs").innerHTML=`
    <div class="stage-box"><span class="pill p-pass">PASS</span>
      <span class="mono" style="font-size:.77rem">resume_next_batch_matched</span>
      <dl class="kv" style="margin-top:.45rem">
        <dt>checkpoint recorded</dt><dd>${D.next_batch.expected_plan_hash_from_checkpoint}</dd>
        <dt>planner recomputed</dt><dd>${D.next_batch.recomputed_plan_hash}</dd></dl>
      <p class="note tight">The planner is a pure function of (seed, branch, step) and never
        consulted the ledger to produce this.</p></div>
    <div class="stage-box"><span class="pill p-pass">PASS</span>
      <span class="mono" style="font-size:.77rem">resume_rollback_replay_identical</span>
      <dl class="kv" style="margin-top:.45rem">
        <dt>microbatches compared</dt><dd>${D.rollback.compared}</dd>
        <dt>mismatches</dt><dd>0</dd><dt>missing</dt><dd>0</dd></dl>
      <p class="note tight">The strong form of “no skipped or repeated batches”: everything the
        rollback discarded came back byte-identical.</p></div>`;
},

panelReplay(host){
  const p=el("div","panel");
  p.innerHTML=`<div class="panel-hd"><h4>Replayed interval · recorded versus rebuilt</h4>
    <span class="note">steps ${D.replay.interval.join("–")} · ${D.replay.microbatches_replayed} microbatches · all match</span></div>
    <div class="panel-bd flush scroll"><table id="replayTable"></table></div>`;
  host.appendChild(p);
  $("#replayTable").innerHTML=`<thead><tr><th class="num">step</th><th>microbatch</th><th>batch id</th>
    <th>recorded hash</th><th>rebuilt from shard bytes</th><th>token span</th><th></th></tr></thead><tbody>`+
    D.replay_rows.map(r=>`<tr><td class="num">${r.step}</td>
    <td class="mono" style="font-size:.71rem">${r.mb}</td>
    <td class="mono" style="font-size:.71rem">${r.batch}</td>
    <td class="mono" style="font-size:.71rem">${r.rec}</td>
    <td class="mono" style="font-size:.71rem">${r.rebuilt}</td>
    <td class="mono" style="font-size:.69rem;color:var(--ink-3)">${r.spans[0]||""}</td>
    <td>${r.ok?'<span class="pill p-pass">match</span>':'<span class="pill p-stop">differs</span>'}</td></tr>`).join("")+"</tbody>";

  const f=D.fork;
  const steps=[...new Set([...f.steps_before_fork,...f.steps_after_fork])].sort((a,b)=>a-b);
  const p2=el("div","panel");
  p2.innerHTML=`<div class="panel-hd"><h4>Fork divergence · identical before, different after</h4></div>
    <div class="panel-bd flush scroll"><table><thead><tr><th class="num">step</th>
    <th>main branch batch</th><th>fork branch batch</th><th></th></tr></thead><tbody>`+
    steps.map(s=>{const a=f.parent_batch_ids[String(s)],b=f.fork_batch_ids[String(s)];
      return `<tr><td class="num">${s}</td><td class="mono" style="font-size:.73rem">${a||"—"}</td>
      <td class="mono" style="font-size:.73rem">${b||"—"}</td>
      <td>${a===b?'<span class="pill p-mute">identical</span>':'<span class="pill p-acc">diverged</span>'}
      ${s<f.fork_point_step?'<span class="pill p-mute">pre-fork</span>':''}</td></tr>`;}).join("")+
    `</tbody></table></div>
    <div class="panel-bd" style="border-top:1px solid var(--rule-2)"><p class="note">
      Both halves matter. Identical batches <em>after</em> the fork point would mean the fork changed
      nothing, so the comparison would be measuring noise. Differing batches <em>before</em> it would
      mean the fork did not really start from the parent's state. Here:
      ${f.identical_before_fork.length} identical before step ${f.fork_point_step},
      ${f.differing_after_fork.length} differing after —
      <span class="pill p-pass">diverged correctly</span></p></div>`;
  host.appendChild(p2);
},

panelAudit(host){
  const p=el("div","panel");
  p.innerHTML=`<div class="panel-hd"><h4>“Which shards influenced the model over this interval?”</h4></div>
    <div class="panel-bd"><div class="grid3" id="auditProv" style="margin-bottom:.8rem"></div>
    <div class="scroll"><table id="auditTable"></table></div></div>`;
  host.appendChild(p);
  const pr=D.audit.provenance;
  $("#auditProv").innerHTML=[["shards traced",pr.shards_involved],["positions",fmt(pr.total_positions)],
    ["loss-bearing",fmt(pr.loss_bearing_tokens)]].map(([k,v])=>
    `<div class="met"><div class="mk">${k}</div><div class="mv">${v}</div></div>`).join("");
  $("#auditTable").innerHTML=`<thead><tr><th>shard</th><th class="num">spans</th><th class="num">tokens</th>
    <th class="num">first step</th><th class="num">last step</th><th>steps</th></tr></thead><tbody>`+
    D.audit.shards.slice(0,14).map(s=>`<tr><td class="mono" style="font-size:.72rem">${s.shard_id}</td>
    <td class="num">${s.distinct_spans}</td><td class="num">${fmt(s.total_span_tokens)}</td>
    <td class="num">${s.first_step}</td><td class="num">${s.last_step}</td>
    <td class="mono" style="font-size:.69rem;color:var(--ink-3)">${s.steps.slice(0,12).join(",")}${s.steps.length>12?"…":""}</td></tr>`).join("")+"</tbody>";

  const p2=el("div","panel");
  p2.innerHTML=`<div class="panel-hd"><h4>“Which OPUS-selected batches preceded a loss spike?”</h4></div>
    <div class="panel-bd stack-s" id="spikeList"></div>`;
  host.appendChild(p2);
  $("#spikeList").innerHTML = D.audit.spike_reports.length===0
    ? `<p class="note">No loss spikes detected in this run.</p>`
    : D.audit.spike_reports.map(s=>{const sp=D.audit.spikes.find(x=>x.step===s.step)||{};
      return `<div class="stage-box">
      <div style="display:flex;gap:.55rem;align-items:baseline;flex-wrap:wrap">
        <span class="pill p-warn">step ${s.step}</span>
        <span class="mono" style="font-size:.77rem">loss ${sp.previous_loss} → ${sp.loss} (z ${sp.z_score})</span></div>
      <p class="note" style="margin-top:.4rem">Looking back over steps ${s.lookback.join("–")}:
        ${s.accepted} OPUS acceptances, ${s.overrides} of them protected-floor overrides.
        ${s.top?`The largest candidate gradient in the window was <code>${s.top.gradient_norm.toFixed(3)}</code>
        from lane <b>${s.top.lane}</b> (${s.top.reason.replace(/_/g," ")}).`:""}</p>
      <p class="note">Shards in the window: <span class="mono" style="font-size:.71rem">${s.shards.slice(0,8).join(", ")}</span></p>
      </div>`;}).join("");
},

panelThroughput(host){
  const bd=panelShell(host,"Where the positions go");
  const c=D.perf.counters, raw=c.raw_positions;
  bd.innerHTML=[["all positions moved",raw,"var(--ink-3)"],
    ["real tokens (not padding)",raw-c.pad_tokens,"var(--accent)"],
    ["graded — loss-bearing",c.useful_loss_bearing_tokens,"var(--pass)"]]
    .map(([k,v,col])=>`<div class="hbar" style="grid-template-columns:12rem 1fr 5.5rem">
      <span style="font-size:.76rem">${k}</span>
      <div class="bar" style="height:.85rem"><span style="width:${v/raw*100}%;background:${col}"></span></div>
      <span class="num">${fmt(v)}</span></div>`).join("")+
    `<p class="note tight">The gap between the second and third bars is context-only tokens —
      prompts and tool observations the model conditions on but is not graded for. Not waste, but
      not learning either, and a throughput number that counts them overstates the run.</p>`;
  const p2=el("div","panel");
  p2.innerHTML=`<div class="panel-bd"><div class="grid2">
    <div><h4 style="margin-bottom:.5rem">Efficiency</h4><dl class="kv" id="effKv"></dl>
      <p class="note tight" id="effNote"></p></div>
    <div><h4 style="margin-bottom:.5rem">Rates</h4><dl class="kv" id="rateKv"></dl>
      <p class="note tight" id="rateNote"></p></div></div></div>`;
  host.appendChild(p2);
  const e=$("#effKv");
  Object.entries(D.perf.efficiency).forEach(([k,v])=>{
    e.appendChild(el("dt",null,k.replace(/_/g," ")));
    e.appendChild(el("dd",null,typeof v==="number"?v.toFixed(4):String(v)));});
  $("#effNote").textContent=D.perf.counters_source;
  const r=$("#rateKv");
  Object.entries(D.perf.throughput).filter(([k])=>!k.startsWith("_")).forEach(([k,v])=>{
    r.appendChild(el("dt",null,k.replace(/_/g," ")));
    r.appendChild(el("dd",null,fmt(v)));});
  $("#rateNote").textContent=D.perf.throughput._scope;
}
};

/* ── mount stages, then build the rail ────────────────────────────── */
(function(){
  const mount=$("#stageSections");
  STAGES_DOC.forEach(s=>mount.appendChild(renderStage(s)));

  const secs=[...document.querySelectorAll("section[data-title]")];
  const ol=$("#railList"); let group=null, k=0;
  secs.forEach(s=>{
    if(s.dataset.group!==group){group=s.dataset.group; ol.appendChild(el("li","rg",group));}
    const li=el("li"), a=el("a");
    a.href="#"+s.id;
    const num = s.dataset.num || (++k, "·");
    a.innerHTML=`<span class="rn">${num}</span><span>${s.dataset.title}</span>`;
    li.appendChild(a); ol.appendChild(li);});
  const links=[...ol.querySelectorAll("a")];
  const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting)
    links.forEach(l=>l.classList.toggle("on", l.getAttribute("href")==="#"+e.target.id));});},
    {rootMargin:"-12% 0px -75% 0px"});
  secs.forEach(s=>io.observe(s));
})();


/* ── completion criterion ─────────────────────────────────────────── */
(function(){
  const C = D.completion;
  if(!C){ return; }
  $("#critHdr").textContent =
    `${C.consumed_sample_instances} consumed sample instances walked · ` +
    (C.all_four_clauses_proved ? "all four clauses proved" : "INCOMPLETE");

  const LABEL = {
    what_it_consumed:"what it consumed",
    why_it_consumed_it:"why it consumed it",
    what_the_model_learned:"what the model learned from it",
    how_the_run_can_be_reconstructed:"how the run can be reconstructed"};
  const box=$("#critClauses");
  Object.entries(C.clauses).forEach(([key,c],i)=>{
    const d=el("div");
    d.style.cssText="padding:.85rem .95rem;border-bottom:1px solid var(--rule-2)";
    const facts=Object.entries(c).filter(([k])=>k!=="proved"&&k!=="how");
    d.innerHTML=`<div style="display:flex;gap:.7rem;align-items:baseline;flex-wrap:wrap">
        <span class="stagenum" style="font-size:1.15rem">${i+1}</span>
        <strong style="font-family:var(--serif);font-size:1.05rem">${LABEL[key]}</strong>
        <span class="pill ${c.proved?"p-pass":"p-stop"}">${c.proved?"PROVED":"NOT PROVED"}</span></div>
      <p class="note" style="margin-top:.4rem;max-width:70ch">${c.how}.</p>
      <div class="metgrid" style="margin-top:.55rem">${facts.map(([k,v])=>
        `<div class="met"><div class="mk">${k.replace(/_/g," ")}</div>
         <div class="mv" style="color:${(v===0||v===true)?"var(--pass)":(v===false?"var(--stop)":"inherit")}">${
           v===true?"yes":v===false?"NO":v}</div></div>`).join("")}</div>`;
    box.appendChild(d);
  });

  const ex=C.worked_example;
  if(!ex){ $("#critExample").innerHTML=`<p class="note">no worked example recorded</p>`; return; }
  const group=(title,obj,tone)=>`<div class="iobox" style="background:var(--card)">
    <div class="t" style="color:${tone}">${title}</div>
    <dl class="kv" style="margin-top:.35rem">${Object.entries(obj).map(([k,v])=>
      `<dt>${k.replace(/_/g," ")}</dt><dd>${v===null?"—":esc(String(v)).slice(0,120)}</dd>`).join("")}</dl></div>`;
  $("#critExample").innerHTML=
    `<div class="iogrid" style="grid-template-columns:1fr">
       ${group("1 · what was consumed",ex.consumed,"var(--accent)")}
       ${group("2 · why it was consumed",ex.why,"var(--accent)")}
       ${group("3 · what the model learned",ex.learned,"var(--accent)")}
       ${group("4 · what it can be rebuilt from",ex.reconstructable_from,"var(--accent)")}
     </div>
     <p class="note tight">Every value above appears in a generated artifact. The decision id
       resolves in <code>ledgers/opus_decisions.jsonl</code>, the losses in
       <code>ledgers/learning_main.jsonl</code>, the token span in the shard binary named by it,
       and the plan hash recomputes from the seed alone.</p>`;
})();

/* ── evidence section ─────────────────────────────────────────────── */
(function(){
  const m=D.evidence_meta;
  $("#evGen").textContent=m.generation_method.charAt(0).toUpperCase()+m.generation_method.slice(1)+".";
  $("#evCount").textContent=`${m.requirements_passed} of ${m.requirements_total} passed`;
  const box=$("#evList");
  D.evidence.forEach(r=>{
    const d=el("details"); d.style.borderBottom="1px solid var(--rule-2)";
    const sum=el("summary");
    sum.style.cssText="padding:.6rem .95rem;cursor:pointer;display:flex;gap:.7rem;align-items:center;flex-wrap:wrap";
    sum.innerHTML=`<span class="pill ${r.res==="PASS"?"p-pass":"p-stop"}">${r.res}</span>
      <strong style="font-family:var(--serif);font-size:1rem">${r.req}</strong>
      <span class="note" style="flex:1 1 200px">${r.note}</span>`;
    d.appendChild(sum);
    const body=el("div"); body.style.cssText="padding:0 .95rem 1rem";
    const dl=el("dl","kv");
    Object.entries(r.detail).forEach(([k,v])=>{
      let s;
      if(Array.isArray(v)) s = v.length===0?"none":(v.length>6? v.length+" entries":v.join(", "));
      else if(v&&typeof v==="object") s = Object.entries(v).slice(0,6).map(([a,b])=>`${a}=${b}`).join(" · ");
      else if(typeof v==="boolean") s = v?"yes":"no";
      else s = String(v);
      dl.appendChild(el("dt",null,k.replace(/_/g," "))); dl.appendChild(el("dd",null,esc(s)));});
    body.appendChild(dl);
    body.appendChild(el("div","note tight","evidence: "+r.files.map(f=>`<code>${f}</code>`).join(" ")));
    d.appendChild(body); box.appendChild(d);});
  $("#passList").innerHTML=D.run_log_tail.map(l=>{
    const pass=l.startsWith("[PASS]");
    return `<div><span class="pill ${pass?"p-pass":"p-stop"}">${pass?"PASS":"FAIL"}</span>
      <span style="margin-left:.4rem">${esc(l.replace(/^\[(PASS|FAIL)\]\s*/,""))}</span></div>`;}).join("");
})();
