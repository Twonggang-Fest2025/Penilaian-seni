import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

env.allowLocalModels = false;
env.useBrowserCache = true;

const $ = id => document.getElementById(id);
const status = $("status"), bar = $("bar"), results = $("results"), csvBtn = $("csv");
let selectedFiles = [], outputRows = [];

const RUBRICS = {
X: [
 {q:"Q1", concepts:["seni","karya","ekspresi","budaya","keindahan"], max:20},
 {q:"Q2", concepts:["kesatuan","unity","keseimbangan","balance","irama","ritme","penekanan","proporsi","harmoni","komposisi"], max:20},
 {q:"Q3", concepts:["titik","garis","bidang","bentuk","ruang","warna","tekstur"], max:20},
 {q:"Q4", concepts:["primer","sekunder","tersier","netral","merah","kuning","biru","hijau","oranye","ungu"], max:20},
 {q:"Q5", concepts:["murni","pure","terapan","fungsi","guna","hias","lukisan","patung","kriya","desain"], max:20}
],
XI: [
 {q:"Q1", concepts:["seni","ekspresi","perasaan","gagasan","karya"], max:20},
 {q:"Q2", concepts:["tenaga","ruang","waktu","gerak","level","arah","tempo","dinamika"], max:20},
 {q:"Q3", concepts:["wiraga","wirasa","wirama","gerak","rasa","irama","keselarasan"], max:20},
 {q:"Q4", concepts:["imitatif","meniru","alam","hewan","manusia","imajinatif","khayalan","kreasi"], max:20},
 {q:"Q5", concepts:["tema","eksplorasi","improvisasi","komposisi","evaluasi","gerak","iringan","pola"], max:20}
],
XII: [
 {q:"Q1", concepts:["klasik","tradisional","modern","kontemporer","zaman","perkembangan","budaya"], max:20},
 {q:"Q2", concepts:["tempo","ritme","irama","melodi","harmoni","dinamika","timbre","birama"], max:20},
 {q:"Q3", concepts:["hari musik dunia","melodis","harmonis","ritmis","melodi","harmoni","ritme","iringan"], max:20},
 {q:"Q4", concepts:["cara memainkan","dipukul","dipetik","ditiup","digesek","digoyang","tekan","contoh","instrumen"], max:20},
 {q:"Q5", concepts:["sumber bunyi","idiofon","membranofon","kordofon","aerofon","elektrofon","contoh"], max:20}
]};

let ocr = null;

$("files").addEventListener("change", e => {
  selectedFiles = [...e.target.files];
  $("drop").textContent = `${selectedFiles.length} foto dipilih.`;
});
["dragover","dragenter"].forEach(ev => $("drop").addEventListener(ev,e=>{e.preventDefault()}));
$("drop").addEventListener("drop",e=>{
  e.preventDefault();
  selectedFiles=[...e.dataTransfer.files].filter(f=>f.type.startsWith("image/"));
  $("drop").textContent=`${selectedFiles.length} foto dipilih.`;
});

function setStatus(t,p=null){
  status.textContent=t;
  if(p!==null) bar.style.width=`${Math.max(0,Math.min(100,p))}%`;
}
function clean(s){return (s||"").replace(/\s+/g," ").trim()}
function tokens(s){return clean(s).toLowerCase().replace(/[^a-z0-9À-ÿ\s-]/gi," ").split(/\s+/).filter(Boolean)}

async function loadOCR(){
  if(ocr) return;
  setStatus("Memuat model pembaca tulisan tangan. Pertama kali bisa cukup lama...",5);
  ocr = await pipeline("image-to-text","Xenova/trocr-small-handwritten",{
    dtype:"q8",
    device:"wasm"
  });
}

async function imageToCanvas(file, scale=1.8){
  const bmp=await createImageBitmap(file);
  const c=document.createElement("canvas");
  c.width=Math.min(2200,Math.round(bmp.width*scale));
  c.height=Math.min(3200,Math.round(bmp.height*scale));
  const ctx=c.getContext("2d");
  ctx.drawImage(bmp,0,0,c.width,c.height);
  return c;
}

async function ocrImage(file){
  const c=await imageToCanvas(file);
  const blob=await new Promise(r=>c.toBlob(r,"image/png"));
  const url=URL.createObjectURL(blob);
  try{
    const out=await ocr(url,{max_new_tokens:180});
    return clean(out?.[0]?.generated_text||"");
  }finally{URL.revokeObjectURL(url)}
}

function splitQuestions(text){
  // Heuristics: OCR may or may not preserve "1.", "2.", etc.
  const lines=(text||"").split(/\n+/).map(clean).filter(Boolean);
  const q=[[],[],[],[],[]]; let current=-1;
  for(const line of lines){
    const m=line.match(/^(?:soal\s*)?([1-5])\s*[\).:\-]/i);
    if(m) current=Number(m[1])-1;
    if(current>=0) q[current].push(line.replace(/^(?:soal\s*)?[1-5]\s*[\).:\-]\s*/i,""));
  }
  if(q.every(x=>x.length===0)){
    // Fallback: divide by approximate text length.
    const words=(text||"").split(/\s+/).filter(Boolean);
    const n=Math.max(1,Math.ceil(words.length/5));
    for(let i=0;i<5;i++) q[i]=words.slice(i*n,(i+1)*n);
  }
  return q.map(x=>clean(x.join(" ")));
}

function scoreAnswer(answer,rubric){
  const t=tokens(answer);
  if(t.length<4) return {score:0,confidence:0,reason:"Tulisan yang terbaca terlalu sedikit."};
  const joined=t.join(" ");
  let hits=0;
  const found=[];
  for(const c of rubric.concepts){
    const ct=tokens(c);
    const phrase=ct.join(" ");
    if(phrase && (ct.length===1 ? t.includes(ct[0]) : joined.includes(phrase))){
      hits++; found.push(c);
    }
  }
  const coverage=Math.min(1,hits/Math.max(3,Math.ceil(rubric.concepts.length*.45)));
  const lengthFactor=Math.min(1,t.length/35);
  const confidence=Math.round((coverage*.7+lengthFactor*.3)*100);
  let score=Math.round(20*coverage);
  // Avoid awarding a high score from a single keyword.
  if(hits<=1) score=Math.min(score,7);
  if(t.length<8) score=Math.min(score,8);
  const reason=found.length
    ? `Konsep terdeteksi: ${found.slice(0,8).join(", ")}.`
    : "Belum ditemukan konsep kunci yang cukup.";
  return {score,confidence,reason};
}

function extractName(text){
  const lines=(text||"").split(/\n+/).map(clean).filter(Boolean);
  for(const l of lines){
    const m=l.match(/(?:nama|nama peserta|nama siswa)\s*[:\-]\s*(.+)/i);
    if(m && m[1].length>2) return m[1].slice(0,80);
  }
  return "Tanpa Nama";
}

function renderRow(r,i){
  return `<div class="card">
    <div class="grid"><div><b>${r.name}</b><div class="small">${r.file}</div></div>
    <div><span class="score">${r.total}/100</span><div class="${r.status==="Selesai"?"ok":"warn"}">${r.status}</div></div></div>
    <table><thead><tr><th>Soal</th><th>Nilai</th><th>Jawaban terbaca</th><th>Keterangan</th></tr></thead><tbody>
    ${r.questions.map(x=>`<tr><td>${x.q}</td><td><b>${x.nilai}</b>/20</td><td class="answer">${escapeHtml(x.jawaban||"")}</td><td>${escapeHtml(x.alasan)}<br><span class="small">Kepercayaan: ${x.confidence}%</span></td></tr>`).join("")}
    </tbody></table>
  </div>`;
}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[m]))}

$("start").addEventListener("click", async()=>{
  if(!selectedFiles.length){setStatus("Pilih minimal satu foto terlebih dahulu.");return}
  $("start").disabled=true; csvBtn.disabled=true; outputRows=[]; results.innerHTML="";
  try{
    await loadOCR();
    const subject=$("subject").value, rub=RUBRICS[subject];
    for(let i=0;i<selectedFiles.length;i++){
      const file=selectedFiles[i];
      setStatus(`Membaca ${i+1}/${selectedFiles.length}: ${file.name}`,10+(i/selectedFiles.length)*80);
      const text=await ocrImage(file);
      const answers=splitQuestions(text);
      const questions=answers.map((a,j)=>({q:`Q${j+1}`,jawaban:a,...scoreAnswer(a,rub[j])}));
      const total=questions.reduce((s,x)=>s+x.score,0);
      const low=questions.filter(x=>x.confidence<35||x.jawaban.length<8).length;
      const row={name:extractName(text),file:file.name,total,status:low>=2?"PERLU CEK":"Selesai",questions,ocr:text};
      outputRows.push(row); results.insertAdjacentHTML("beforeend",renderRow(row,i));
    }
    setStatus("Selesai. Hasil di bawah dapat diperiksa dan diekspor.",100);
    csvBtn.disabled=false;
  }catch(err){
    console.error(err);
    setStatus("Terjadi kesalahan: "+(err?.message||err));
  }finally{$("start").disabled=false}
});

csvBtn.addEventListener("click",()=>{
  const rows=[["Nama","File","Q1","Q2","Q3","Q4","Q5","Total","Status"]];
  for(const r of outputRows) rows.push([r.name,r.file,...r.questions.map(q=>q.nilai),r.total,r.status]);
  const csv=rows.map(row=>row.map(v=>`"${String(v??"").replaceAll('"','""')}"`).join(",")).join("\n");
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob(["\ufeff"+csv],{type:"text/csv;charset=utf-8"}));
  a.download="hasil_nilaikita.csv"; a.click(); URL.revokeObjectURL(a.href);
});
