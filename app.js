import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

const $ = id => document.getElementById(id);
let ocr = null, allResults = [];

const RUBRICS = {
  rupa: [
    {q:"Q1", max:20, concepts:["seni","karya","ekspresi","budaya","perasaan","gagasan"], cues:["ungkapan","manusia","keindahan","kreativitas"]},
    {q:"Q2", max:20, concepts:["kesatuan","keseimbangan","irama","penekanan","proporsi","keselarasan","komposisi"]},
    {q:"Q3", max:20, concepts:["titik","garis","bidang","bentuk","ruang","warna","tekstur"]},
    {q:"Q4", max:20, concepts:["primer","sekunder","tersier","netral","merah","kuning","biru","hijau","oranye","ungu"]},
    {q:"Q5", max:20, concepts:["murni","terapan","fungsi","keindahan","kriya","desain","lukisan","kerajinan"]}
  ],
  tari: [
    {q:"Q1", max:20, concepts:["seni","ekspresi","keindahan","gagasan","perasaan"]},
    {q:"Q2", max:20, concepts:["tenaga","ruang","waktu","kuat","lemah","arah","level","tempo","ritme"]},
    {q:"Q3", max:20, concepts:["wiraga","wirasa","wirama","gerak","rasa","irama"]},
    {q:"Q4", max:20, concepts:["imitatif","imajinatif","meniru","hewan","alam","khayalan","gagasan"]},
    {q:"Q5", max:20, concepts:["tema","eksplorasi","improvisasi","komposisi","evaluasi","gerak","iringan"]}
  ],
  musik: [
    {q:"Q1", max:20, concepts:["klasik","tradisional","modern","kontemporer","zaman","budaya","teknologi"]},
    {q:"Q2", max:20, concepts:["tempo","ritme","melodi","harmoni","dinamika","timbre","birama"]},
    {q:"Q3", max:20, concepts:["melodis","harmonis","ritmis","melodi","akor","irama","ritme"]},
    {q:"Q4", max:20, concepts:["cara memainkan","dipukul","ditiup","dipetik","digesek","ditekan","digetarkan"]},
    {q:"Q5", max:20, concepts:["idiofon","membranofon","kordofon","aerofon","elektrofon","sumber bunyi"]}
  ]
};

function normalize(s){
  return s.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g,"")
    .replace(/[^a-z0-9\s.,!?-]/g," ").replace(/\s+/g," ").trim();
}
function esc(s){return (s||"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}

async function loadOCR(){
  if(ocr) return;
  setStatus("Memuat mesin OCR tulisan tangan. Pada pemakaian pertama proses ini bisa cukup lama...");
  ocr = await pipeline("image-to-text","Xenova/trocr-base-handwritten",{device:"wasm"});
}

function setStatus(s){$("status").textContent=s}
function progress(v){$("bar").style.width=Math.max(0,Math.min(100,v))+"%"}

function imageFromFile(file){
  return new Promise((resolve,reject)=>{
    const img=new Image(); img.onload=()=>resolve(img); img.onerror=reject;
    img.src=URL.createObjectURL(file);
  });
}

function canvasFromImage(img,maxSide=1800){
  const scale=Math.min(1,maxSide/Math.max(img.naturalWidth||img.width,img.naturalHeight||img.height));
  const c=document.createElement("canvas"); c.width=Math.round(img.width*scale); c.height=Math.round(img.height*scale);
  c.getContext("2d").drawImage(img,0,0,c.width,c.height); return c;
}

/* Detect horizontal answer lines. The form's printed ruled lines make this much
   more stable than sending the whole page to TrOCR. */
function lineBoxes(canvas){
  const ctx=canvas.getContext("2d"), w=canvas.width,h=canvas.height;
  const data=ctx.getImageData(0,0,w,h).data;
  const rows=[]; const step=Math.max(2,Math.floor(w/900));
  for(let y=Math.floor(h*.14);y<Math.floor(h*.97);y+=2){
    let dark=0, total=0;
    for(let x=Math.floor(w*.08);x<Math.floor(w*.93);x+=step){
      const i=(y*w+x)*4, g=(data[i]+data[i+1]+data[i+2])/3;
      if(g<155) dark++; total++;
    }
    rows.push([y,dark/Math.max(1,total)]);
  }
  const peaks=rows.filter((r,i)=>r[1]>.16 && (i===0||r[1]>=rows[i-1][1]) && (i===rows.length-1||r[1]>=rows[i+1][1])).map(x=>x[0]);
  const grouped=[];
  for(const y of peaks){
    if(!grouped.length||y-grouped[grouped.length-1]>7) grouped.push(y);
    else grouped[grouped.length-1]=(grouped[grouped.length-1]+y)/2;
  }
  const boxes=[];
  for(let i=0;i<grouped.length-1;i++){
    const y1=grouped[i]+4, y2=grouped[i+1]-3;
    if(y2-y1<8||y2-y1>100) continue;
    boxes.push({x:Math.floor(w*.10),y:y1,w:Math.floor(w*.82),h:y2-y1});
  }
  return boxes.slice(0,80);
}

function cropLine(canvas,b){
  const c=document.createElement("canvas"); c.width=b.w; c.height=Math.max(32,b.h);
  const ctx=c.getContext("2d"); ctx.fillStyle="#fff"; ctx.fillRect(0,0,c.width,c.height);
  ctx.drawImage(canvas,b.x,b.y,b.w,b.h,0,0,c.width,c.height);
  return c;
}

function cleanLine(c){
  const ctx=c.getContext("2d"), img=ctx.getImageData(0,0,c.width,c.height), d=img.data;
  for(let i=0;i<d.length;i+=4){
    const g=.299*d[i]+.587*d[i+1]+.114*d[i+2];
    const v=g<185?20:255;
    d[i]=d[i+1]=d[i+2]=v;
  }
  ctx.putImageData(img,0,0);
  return c;
}

async function ocrLine(c){
  const out=await ocr(c,{max_new_tokens:80,num_beams:2});
  return (out?.[0]?.generated_text||"").trim();
}

function splitQuestions(lines){
  const q=[[],[],[],[],[]];
  let current=-1;
  for(const raw of lines){
    const t=raw.trim(); if(!t) continue;
    const n=t.match(/^\s*([1-5])(?:\s*[\.\):\-]|$)/);
    if(n) current=Number(n[1])-1;
    if(current>=0) q[current].push(t);
  }
  return q.map(a=>a.join(" "));
}

function scoreAnswer(text,rubric){
  const s=normalize(text);
  if(s.length<8) return {nilai:0,alasan:"Jawaban terlalu sedikit atau belum terbaca jelas.",confidence:0.05};
  let hits=0;
  const found=[];
  for(const c of rubric.concepts){
    if(s.includes(normalize(c))){hits++;found.push(c);}
  }
  const words=new Set(s.split(/\s+/));
  const density=Math.min(1,words.size/55);
  const coverage=Math.min(1,hits/Math.max(3,Math.min(7,rubric.concepts.length)));
  let score=Math.round((coverage*.72+density*.28)*20);
  if(score>20)score=20;
  let confidence=Math.min(.98,.25+coverage*.6+density*.2);
  if(s.length<30) confidence*=.8;
  const reason=found.length?`Konsep terdeteksi: ${found.slice(0,6).join(", ")}.`:"Belum ditemukan konsep kunci yang cukup.";
  return {nilai:score,alasan:reason,confidence};
}

function chooseRubric(text){
  const s=normalize(text);
  const sets=Object.entries(RUBRICS).map(([name,rs])=>{
    const hits=rs.flatMap(r=>r.concepts).filter(c=>s.includes(normalize(c))).length;
    return [name,hits];
  }).sort((a,b)=>b[1]-a[1]);
  return sets[0][1]?sets[0][0]:"rupa";
}

async function processFile(file,index,total){
  const img=await imageFromFile(file);
  const canvas=canvasFromImage(img);
  const boxes=lineBoxes(canvas);
  const lines=[];
  for(let i=0;i<boxes.length;i++){
    const c=cleanLine(cropLine(canvas,boxes[i]));
    const text=await ocrLine(c);
    if(text && text.replace(/\W/g,"").length>=2) lines.push(text);
    progress(((index+(i+1)/Math.max(1,boxes.length))/total)*100);
  }
  const combined=lines.join(" ");
  const rubricName=chooseRubric(combined);
  const answers=splitQuestions(lines);
  const scores=RUBRICS[rubricName].map((r,i)=>scoreAnswer(answers[i],r));
  const totalScore=scores.reduce((a,b)=>a+b.nilai,0);
  const low=scores.filter(x=>x.confidence<.5).length;
  return {file:file.name,rubric:rubricName,answers,scores,total:totalScore,status:low>=2?"PERLU CEK":"SELESAI"};
}

function render(){
  $("results").innerHTML=allResults.map((r,idx)=>`
  <div class="result">
    <h3>${idx+1}. ${esc(r.file)}</h3>
    <div><span class="pill">${r.rubric}</span><span class="${r.status==="SELESAI"?"ok":"warn"}">${r.status}</span></div>
    <div class="score">${r.total}/100</div>
    <table><thead><tr><th>Soal</th><th>Nilai</th><th>Hasil baca</th><th>Alasan</th></tr></thead>
    <tbody>${r.scores.map((s,i)=>`<tr><td>Q${i+1}</td><td><b>${s.nilai}/20</b></td><td>${esc(r.answers[i]||"—")}</td><td>${esc(s.alasan)}</td></tr>`).join("")}</tbody></table>
  </div>`).join("");
}

function csv(){
  const rows=[["File","Rubrik","Status","Q1","Q2","Q3","Q4","Q5","Total"]];
  for(const r of allResults) rows.push([r.file,r.rubric,r.status,...r.scores.map(s=>s.nilai),r.total]);
  const blob=new Blob([rows.map(row=>row.map(v=>`"${String(v).replaceAll('"','""')}"`).join(",")).join("\n")],{type:"text/csv;charset=utf-8"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="hasil-nilaikita.csv";a.click();
}

$("start").onclick=async()=>{
  const files=[...$("files").files].slice(0,10);
  if(!files.length){setStatus("Pilih minimal satu foto.");return;}
  $("start").disabled=true;$("download").disabled=true;allResults=[];render();progress(0);
  try{
    await loadOCR();
    for(let i=0;i<files.length;i++){
      setStatus(`Mengerjakan foto ${i+1} dari ${files.length}...\nMendeteksi baris dan membaca tulisan tangan.`);
      allResults.push(await processFile(files[i],i,files.length)); render();
    }
    setStatus(`Selesai memproses ${files.length} foto.`);
    $("download").disabled=false;
  }catch(e){
    console.error(e); setStatus("Terjadi kesalahan: "+(e?.message||e));
  }finally{$("start").disabled=false;}
};
$("download").onclick=csv;
