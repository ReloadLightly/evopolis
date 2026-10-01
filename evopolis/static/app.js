import {drawTown, hitResident} from './town.js';

const $ = (selector, parent = document) => parent.querySelector(selector);
const $$ = (selector, parent = document) => [...parent.querySelectorAll(selector)];
const letters = ['A', 'B', 'C', 'D'];
const colors = {Equal:'#ff8b8b', Mixed:'#ffcf6e', Proportional:'#6dcff6', 'Recorded RL M1':'#73e0b3'};
const defaults = [0.75, 0.75, 0.75, 0.25];
const state = {
  mode:'recorded', compare:false, round:0, phase:0, playing:false,
  reduced:matchMedia('(prefers-reduced-motion: reduce)').matches, speed:1,
  catalog:null, generation:0, busy:true,
  panes:[0,1].map(i => ({cohort:'human', mechanism:i ? 'Proportional':'Equal',
    id:null, selected:0, rule:i ? 'proportional':'equal', fractions:[...defaults], episode:null, element:null})),
};
const fmt = value => value == null ? 'N/A' : Number(value).toLocaleString('en-US', {maximumFractionDigits:Math.abs(value)<1 ? 5:2});
const exact = value => value == null ? 'Unavailable' : String(value);
const esc = value => String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const svgText = (x,y,text,color='#b6c6df',anchor='start') => `<text x="${x}" y="${y}" fill="${color}" font-family="monospace" font-size="10" text-anchor="${anchor}">${esc(text)}</text>`;
const activePanes = () => state.panes.slice(0, state.compare ? 2:1);
const maxRounds = () => Math.max(1,...activePanes().map(p=>p.episode?.rounds.length || 1));

async function request(path, body) {
  const response = await fetch(path, body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : undefined);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
function showError(error) { $('#error').textContent = error.message; $('#error').hidden = false; pause(); }
function candidates(pane) { return state.catalog.episodes.filter(e=>e.cohort===pane.cohort && e.mechanism===pane.mechanism); }
function medianEpisode(episodes) {
  const ordered=[...episodes].sort((a,b)=>a.surplus-b.surplus);
  const n=ordered.length, low=ordered[Math.floor((n-1)/2)].surplus, high=ordered[Math.floor(n/2)].surplus;
  const key=e=>JSON.stringify([e.condition,e.launch_id,e.episode_id]);
  // The two middle outcomes are equidistant from the exact median. Avoid a
  // floating midpoint that could accidentally override the stable key tie.
  return ordered.filter(e=>e.surplus===low||e.surplus===high).sort((a,b)=>key(a)<key(b)?-1:key(a)>key(b)?1:0)[0];
}
function selectionOptions(pane) {
  const list=candidates(pane), select=$('.episode',pane.element);
  if (!list.some(e=>e.id===pane.id)) pane.id=medianEpisode(list).id;
  select.replaceChildren(...[...list].sort((a,b)=>a.surplus-b.surplus || a.id.localeCompare(b.id)).map(e=>{
    const option=document.createElement('option'); option.value=e.id;
    option.textContent=`${e.launch_id} · ep ${e.episode_id} | ${e.surplus.toFixed(3)} / ${e.gini?.toFixed(3) ?? 'undefined'}`;
    return option;
  }));
  select.value=pane.id;
}
function mountPanes() {
  const container=$('#communities'); container.replaceChildren(); container.classList.toggle('compare',state.compare);
  activePanes().forEach((pane,index)=>{
    const element=$('#community-template').content.firstElementChild.cloneNode(true);
    pane.element=element; element.dataset.pane=index; element.setAttribute('aria-label',`Community ${index+1}`);
    $('.pane-name',element).textContent=`COMMUNITY ${index+1} / ${state.mode==='recorded' ? 'ARCHIVE':'SANDBOX'}`;
    $('.recorded-selectors',element).hidden=state.mode!=='recorded';
    $('.sandbox-selectors',element).hidden=state.mode!=='sandbox';
    $('.outcome-detail',element).hidden=state.mode!=='recorded';
    $('.cohort',element).value=pane.cohort; $('.mechanism',element).value=pane.mechanism;
    selectionOptions(pane);
    $('.cohort',element).addEventListener('change',e=>{pane.cohort=e.target.value;pane.id=null;selectionOptions(pane);loadEpisodes(true);});
    $('.mechanism',element).addEventListener('change',e=>{pane.mechanism=e.target.value;pane.id=null;selectionOptions(pane);loadEpisodes(true);});
    $('.episode',element).addEventListener('change',e=>{pane.id=e.target.value;loadEpisodes(true);});
    $('.sandbox-mechanism',element).value=pane.rule;
    $('.sandbox-mechanism',element).addEventListener('change',e=>{pane.rule=e.target.value;loadEpisodes(true);});
    $('.fractions',element).innerHTML=letters.map((letter,i)=>`<label>${letter}<input type="number" min="0" max="1" step="0.01" value="${pane.fractions[i]}" aria-label="Community ${index+1} resident ${letter} return fraction"></label>`).join('');
    $$('.fractions input',element).forEach((input,i)=>input.addEventListener('change',()=>{
      if (!input.validity.valid || input.value==='') {input.reportValidity();input.value=pane.fractions[i];return;}
      pane.fractions[i]=Number(input.value);loadEpisodes(true);
    }));
    $('.reset-sandbox',element).addEventListener('click',()=>{pane.fractions=[...defaults];pane.rule='equal';mountPanes();loadEpisodes(true);});
    $('.resident-buttons',element).innerHTML=letters.map((letter,i)=>`<button data-resident="${i}" aria-label="Inspect resident ${letter}">Resident ${letter}</button>`).join('');
    $$('.resident-buttons button',element).forEach(button=>button.addEventListener('click',()=>{pane.selected=Number(button.dataset.resident);renderPane(pane);writeURL();}));
    $('.town',element).addEventListener('click',event=>{const i=hitResident(event.target,event);if(i!=null&&i>=0){pane.selected=i;renderPane(pane);writeURL();}});
    container.append(element);
  });
  $('#mode-context').textContent=state.mode==='recorded'
    ? (state.compare ? 'SYNCHRONIZED REPLAY · Different observed groups. These are not individual counterfactuals.' : 'RECORDED EVIDENCE · 160 human communities + 2,048 upstream BC1 games. Every eligible episode is available.')
    : (state.compare ? 'SCRIPTED COMPARISON · Same fractions copied on entry; each pane can be explicitly edited. Outcomes depend on these fixed assumptions.' : 'NEW SCRIPTED SIMULATION · Fixed-fraction returns using the published equation. No human prediction, fitting or learning.');
}
async function loadEpisodes(reset=false) {
  pause(); const generation=++state.generation;
  state.busy=true;$('#loading').hidden=false;$('#error').hidden=true;
  activePanes().forEach(p=>{p.element.classList.add('loading');p.element.setAttribute('aria-busy','true');});
  $('#loading').textContent='Loading community records…';
  if(reset){state.round=0;state.phase=0;}
  renderTransport();
  try {
    const panes=activePanes();
    const episodes=await Promise.all(panes.map(p=>state.mode==='recorded'
      ? request(`/api/episode?id=${encodeURIComponent(p.id)}`)
      : request('/api/sandbox',{mechanism:p.rule,fractions:p.fractions})));
    if(generation!==state.generation)return;
    panes.forEach((pane,i)=>{pane.episode=episodes[i];pane.element.classList.remove('loading');pane.element.setAttribute('aria-busy','false');});
    state.round=Math.min(state.round,maxRounds()-1);state.busy=false;$('#loading').hidden=true;
    render();writeURL();
  } catch(error) {if(generation===state.generation){$('#loading').hidden=true;showError(error);renderTransport();}}
}
function pause() {state.playing=false;$('#play').textContent='▶ Play';$('#play').setAttribute('aria-label','Play');}
function seek(round) {if(state.busy)return;pause();state.round=Math.max(0,Math.min(maxRounds()-1,round));state.phase=0;render();writeURL();}
function renderTransport() {
  $('#round-label').textContent=`${String(state.round+1).padStart(2,'0')} / ${maxRounds()}`;
  $('#scrubber').max=maxRounds()-1;$('#scrubber').value=state.round;
  $('#previous').disabled=state.busy||state.round===0;
  $('#next').disabled=state.busy||state.round>=maxRounds()-1;
  $('#play').disabled=state.busy;$('#restart').disabled=state.busy;$('#scrubber').disabled=state.busy;
}
function drawScene(pane,progress=0) {
  if(!pane.episode)return;
  const finished=state.round>=pane.episode.rounds.length;
  const round=pane.episode.rounds[Math.min(state.round,pane.episode.rounds.length-1)];
  drawTown($('.town',pane.element),{round,phase:finished?2:state.phase,selectedResident:pane.selected,reducedMotion:state.reduced,progress});
}
function cumulativeGini(values) {
  const total=values.reduce((a,b)=>a+b,0);
  return total===0 ? null : values.flatMap(a=>values.map(b=>Math.abs(a-b))).reduce((a,b)=>a+b,0)/(8*total);
}
function renderPane(pane) {
  const el=pane.element, episode=pane.episode;if(!episode)return;
  const selectedIndex=Math.min(state.round,episode.rounds.length-1), r=episode.rounds[selectedIndex], i=pane.selected;
  const isScript=episode.cohort==='scripted', finished=state.round>=episode.rounds.length;
  const source=$('.source-badge',el);source.className=`source-badge ${episode.cohort}`;
  source.textContent=isScript?'NEW SCRIPTED SIMULATION':episode.cohort==='human'?'RECORDED HUMAN · EXP 1':'RECORDED UPSTREAM MODEL · BC1';
  $('.identity',el).textContent=isScript
    ? `${episode.mechanism} · q = [${pane.fractions.join(', ')}] · ${episode.rounds.length} rounds · ${episode.termination}`
    : `${episode.condition} · launch ${episode.launch_id} · episode ${episode.episode_id} · source rounds 0–39`;
  $$('.phase-strip span',el).forEach((span,index)=>span.classList.toggle('active',index===(finished?2:state.phase)));
  $('.pool-before',el).textContent=fmt(r.pool_before);$('.pool-before',el).title=exact(r.pool_before);
  $('.pool-after',el).textContent=fmt(r.pool_after);$('.pool-after',el).title=r.pool_after==null?'No following observation':exact(r.pool_after);
  $('.after-label',el).textContent=isScript?'Simulated pool after':'Recorded pool after';
  $('.active-count',el).textContent=`${r.active_count} / 4`;
  $('.total-surplus',el).textContent=fmt(r.group_cumulative_surplus ?? r.cumulative_surplus.reduce((a,b)=>a+b,0));
  $('.resident-name',el).textContent=`Resident ${letters[i]}`;
  $('.resident-position',el).textContent=`Position ${i} in this group · values in resource units`;
  $$('.resident-buttons button',el).forEach((b,j)=>b.setAttribute('aria-pressed',j===i?'true':'false'));
  $('.resident-values',el).innerHTML=[['Allocation',r.offers[i]],['Returned',r.contributions[i]],['Retained this round',r.surplus[i]],['Cumulative retained',r.cumulative_surplus[i]]].map(([label,value])=>`<div><dt>${label}</dt><dd title="${esc(exact(value))}">${fmt(value)}</dd></div>`).join('');
  const opportunity=$('.opportunity',el);opportunity.classList.toggle('low',r.offers[i]<1);
  opportunity.textContent=r.offers[i]<1?'Offer below 1 unit in this round. This alone does not establish permanent exclusion.':'Offer at least 1 unit in this round.';
  $('.history',el).innerHTML=episode.rounds.slice(0,selectedIndex+1).map(h=>`<tr><td>${h.round_id+1}</td>${[h.offers[i],h.contributions[i],h.surplus[i],h.cumulative_surplus[i]].map(v=>`<td title="${esc(exact(v))}">${fmt(v)}</td>`).join('')}</tr>`).join('');
  const previous=selectedIndex ? episode.rounds[selectedIndex-1] : null;
  $('.information-basis',el).textContent=isScript
    ? 'This fixed script uses only its own allocation and chosen fraction q. Public history below is researcher context, not a script input.'
    :episode.cohort==='bc1'
      ? 'The upstream BC1 model used nine inputs: all current offers, previous returns and current pool, plus recurrent memory. Cumulative totals below are researcher context, not a separate model input.'
      : 'People saw all current offers, previous public returns and pool size; earlier public outcomes and cumulative surplus could be remembered. Current returns were simultaneous.';
  $('.previous-returns',el).textContent=`Previous returns A–D: ${(previous?.contributions||[0,0,0,0]).map(exact).join(' · ')}`;
  $('.prior-cumulative',el).textContent=`Public retained totals before this round A–D: ${(previous?.cumulative_surplus||[0,0,0,0]).map(exact).join(' · ')}`;
  const raw=r.raw||{};
  const rawVector=(name,fallback)=>fallback.map((value,j)=>raw[`${name}_${j}`] ?? exact(value));
  const offers=rawVector('offer',r.offers),returns=rawVector('player_action',r.contributions),surplus=rawVector('player_reward',r.surplus);
  $('.exact-values',el).innerHTML=letters.map((letter,j)=>`<tr><td>${letter}</td><td>${esc(offers[j])}</td><td>${esc(returns[j])}</td><td>${esc(surplus[j])}</td><td>${esc(exact(r.cumulative_surplus[j]))}</td></tr>`).join('');
  $('.exact-pool',el).textContent=`Pool before: ${raw['mechanism_observation.pool'] ?? exact(r.pool_before)} · ${isScript?'Simulated':'Recorded'} after: ${r.pool_after_raw ?? exact(r.pool_after)}`;
  $('.accounting',el).textContent=`Equation estimate: ${exact(r.equation_after)} · ${isScript?'Simulated':'Recorded'} − equation: ${r.pool_after==null?'Unavailable':exact(r.pool_after-r.equation_after)}. Observations are never replaced by this estimate.`;
  $('.after-source',el).textContent=`Next-pool source: ${r.after_source}. ${isScript?'Internal':'Source'} round ${r.round_id}; playback round ${r.round_id+1}.`;
  const gini=cumulativeGini(r.cumulative_surplus);
  $('.gini-note',el).textContent=`Cumulative retained sums ${isScript?'simulated':'recorded'} round surplus through this round${isScript?'':' (not the rounded source cumulative counter)'}. Current-game cumulative Gini: ${gini==null?'Undefined (zero total)':exact(gini)}. Four-player maximum 0.75. Completed-game Gini: ${episode.gini==null?'Undefined (zero total)':exact(episode.gini)}.${raw.players_cumulative_reward ? ` Source cumulative counter A–D: ${raw.players_cumulative_reward}`:''}`;
  $('.end-state',el).textContent=finished?`Run ended after round ${episode.rounds.length} (${episode.termination}); final state held while the other pane continues.`
    :r.pool_after==null?'Final BC1 next pool is unavailable; no later observation was released.'
    :isScript&&selectedIndex===episode.rounds.length-1?`Run ends: ${episode.termination}. No undocumented 0.01 floor.`
    :`Units: flowers / retained resources. All four contributions were simultaneous.`;
  drawTimeline(pane,selectedIndex);
  drawScene(pane);
  if(!isScript)drawScatter(pane);
}
function drawTimeline(pane,index) {
  const rows=pane.episode.rounds, x=j=>38+j*470/Math.max(1,rows.length-1);
  const cumulative=rows.map(r=>r.group_cumulative_surplus ?? r.cumulative_surplus.reduce((a,b)=>a+b,0));
  const maxSurplus=Math.max(1,...cumulative), yPool=v=>100-v/200*78,ySurplus=v=>100-v/maxSurplus*78;
  const path=(values,y)=>values.map((v,j)=>`${j?'L':'M'}${x(j)},${y(v)}`).join(' ');
  const svg=$('.timeline',pane.element);
  svg.innerHTML=`<title>Pool before each round (left axis 0–200); total cumulative retained (right axis 0–${maxSurplus}). Cursor at round ${index+1}.</title><path d="M38 22V100H508" fill="none" stroke="#3b5278"/><path d="M38 61H508 M38 22H508" fill="none" stroke="#253a5e"/>${svgText(31,26,'200','#6dcff6','end')}${svgText(31,104,'0','#6dcff6','end')}${svgText(515,26,Math.round(maxSurplus),'#ffcf6e')}${svgText(515,104,'0','#ffcf6e')}${svgText(38,120,'1')}${svgText(270,120,'Round','#b6c6df','middle')}${svgText(508,120,rows.length,'#b6c6df','end')}<path d="${path(rows.map(r=>r.pool_before),yPool)}" fill="none" stroke="#6dcff6" stroke-width="2"/><path d="${path(cumulative,ySurplus)}" fill="none" stroke="#ffcf6e" stroke-width="2"/><path d="M${x(index)} 15V104" stroke="#fff1d2" stroke-dasharray="3 3"/><circle cx="${x(index)}" cy="${yPool(rows[index].pool_before)}" r="3" fill="#6dcff6"/><circle cx="${x(index)}" cy="${ySurplus(cumulative[index])}" r="3" fill="#ffcf6e"/>`;
}
function drawScatter(pane) {
  const list=candidates(pane),svg=$('.scatter',pane.element), color=colors[pane.mechanism];
  $('.selection-rule',pane.element).textContent='Default within each condition: episode nearest the median mean surplus, with ties broken by the full episode key. All games are shown; selection is not a representative causal comparison.';
  svg.innerHTML=`<path d="M40 15V155H520" fill="none" stroke="#3b5278"/>${svgText(35,20,'14','#b6c6df','end')}${svgText(35,158,'0','#b6c6df','end')}${svgText(40,170,'0')}${svgText(520,170,'0.75','#b6c6df','end')}${svgText(285,185,'Completed-game Gini','#b6c6df','middle')}${svgText(45,12,'Mean surplus / player / round')}`;
  for(const e of list){
    const x=40+e.gini/0.75*480,y=155-e.surplus/14*130,r=e.id===pane.id?6:3;
    const mark=document.createElementNS('http://www.w3.org/2000/svg',pane.mechanism==='Equal'?'circle':'path');
    if(pane.mechanism==='Equal') {mark.setAttribute('cx',x);mark.setAttribute('cy',y);mark.setAttribute('r',r);}
    else {
      const paths={Mixed:`M${x-r} ${y-r}h${2*r}v${2*r}h${-2*r}Z`,Proportional:`M${x} ${y-r*1.2}L${x+r} ${y+r}H${x-r}Z`,'Recorded RL M1':`M${x} ${y-r*1.3}L${x+r} ${y}L${x} ${y+r*1.3}L${x-r} ${y}Z`};
      mark.setAttribute('d',paths[pane.mechanism]);
    }
    mark.setAttribute('fill',color);mark.setAttribute('opacity',e.id===pane.id?1:0.65);mark.setAttribute('stroke',e.id===pane.id?'#fff1d2':'none');
    mark.setAttribute('tabindex','0');mark.setAttribute('role','button');
    mark.setAttribute('aria-label',`Episode ${e.launch_id}, ${e.episode_id}; surplus ${e.surplus.toFixed(3)}, Gini ${e.gini.toFixed(3)}`);
    const choose=()=>{pane.id=e.id;$('.episode',pane.element).value=e.id;loadEpisodes(true);};
    mark.addEventListener('click',choose);mark.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();choose();}});
    svg.append(mark);
  }
}
function render() {renderTransport();activePanes().forEach(renderPane);}
function writeURL() {
  if(state.busy||!state.catalog)return;
  const params=new URLSearchParams({mode:state.mode,compare:state.compare?'1':'0',round:String(state.round+1)});
  activePanes().forEach((p,i)=>{
    params.set(`resident${i}`,String(p.selected));
    if(state.mode==='recorded')params.set(`episode${i}`,p.id);
    else {params.set(`rule${i}`,p.rule);params.set(`q${i}`,p.fractions.join(','));}
  });
  history.replaceState(null,'',`?${params}`);
}
function setMode(mode) {
  if(mode===state.mode)return;pause();state.mode=mode;
  if(mode==='sandbox'&&state.compare)state.panes[1].fractions=[...state.panes[0].fractions];
  $('#recorded-mode').classList.toggle('active',mode==='recorded');$('#sandbox-mode').classList.toggle('active',mode==='sandbox');
  $('#recorded-mode').setAttribute('aria-pressed',mode==='recorded');$('#sandbox-mode').setAttribute('aria-pressed',mode==='sandbox');
  mountPanes();loadEpisodes(true);
}
$('#recorded-mode').addEventListener('click',()=>setMode('recorded'));
$('#sandbox-mode').addEventListener('click',()=>setMode('sandbox'));
$('#compare').addEventListener('change',e=>{pause();state.compare=e.target.checked;if(state.compare&&state.mode==='sandbox')state.panes[1].fractions=[...state.panes[0].fractions];mountPanes();loadEpisodes(true);});
$('#reduced-motion').checked=state.reduced;
$('#reduced-motion').addEventListener('change',e=>{state.reduced=e.target.checked;render();});
$('#restart').addEventListener('click',()=>seek(0));$('#previous').addEventListener('click',()=>seek(state.round-1));$('#next').addEventListener('click',()=>seek(state.round+1));
$('#scrubber').addEventListener('input',e=>seek(Number(e.target.value)));
$('#speed').addEventListener('change',e=>{state.speed=Number(e.target.value);phaseStart=performance.now();});
let phaseStart=0;
$('#play').addEventListener('click',()=>{
  if(state.playing){pause();writeURL();return;}
  if(state.round===maxRounds()-1&&state.phase===2){state.round=0;state.phase=0;}
  state.playing=true;phaseStart=performance.now();$('#play').textContent='Ⅱ Pause';$('#play').setAttribute('aria-label','Pause');render();
});
$('#copy-link').addEventListener('click',async()=>{writeURL();try{await navigator.clipboard.writeText(location.href);$('#copy-link').textContent='Link copied';setTimeout(()=>$('#copy-link').textContent='Copy view link',1800);}catch{showError(new Error('Copy the current address from your browser to share this view.'));}});
document.addEventListener('keydown',e=>{
  if(['INPUT','SELECT','TEXTAREA','BUTTON','SUMMARY'].includes(e.target.tagName)||e.target.getAttribute('role')==='button')return;
  if(e.code==='Space'){e.preventDefault();$('#play').click();}
  if(e.key==='ArrowRight'){e.preventDefault();seek(state.round+1);}if(e.key==='ArrowLeft'){e.preventDefault();seek(state.round-1);}
});
function tick(now){
  if(state.playing&&!state.busy){
    const duration=850/state.speed, elapsed=now-phaseStart;
    if(elapsed>=duration){phaseStart=now;if(state.phase<2)state.phase++;else if(state.round<maxRounds()-1){state.round++;state.phase=0;}else{pause();writeURL();}render();}
    else activePanes().forEach(p=>drawScene(p,elapsed/duration));
  }
  requestAnimationFrame(tick);
}
async function init(){
  try{
    state.catalog=await request('/api/catalog');state.panes[0].id=state.catalog.default_id;
    const params=new URLSearchParams(location.search);
    state.compare=params.get('compare')==='1';$('#compare').checked=state.compare;
    const initialMode=params.get('mode')==='sandbox'?'sandbox':'recorded';
    state.panes.forEach((pane,i)=>{
      const saved=state.catalog.episodes.find(e=>e.id===params.get(`episode${i}`));
      if(saved){pane.id=saved.id;pane.cohort=saved.cohort;pane.mechanism=saved.mechanism;}
      const resident=Number(params.get(`resident${i}`));if(Number.isInteger(resident)&&resident>=0&&resident<4)pane.selected=resident;
      const rule=params.get(`rule${i}`);if(['equal','mixed','proportional','interpolating'].includes(rule))pane.rule=rule;
      const fractions=params.get(`q${i}`)?.split(',').map(Number);if(fractions?.length===4&&fractions.every(v=>Number.isFinite(v)&&v>=0&&v<=1))pane.fractions=fractions;
    });
    const round=Number(params.get('round'));if(Number.isInteger(round)&&round>=1&&round<=40)state.round=round-1;
    state.mode=initialMode;
    $('#recorded-mode').classList.toggle('active',initialMode==='recorded');$('#sandbox-mode').classList.toggle('active',initialMode==='sandbox');
    $('#recorded-mode').setAttribute('aria-pressed',initialMode==='recorded');$('#sandbox-mode').setAttribute('aria-pressed',initialMode==='sandbox');
    $('#provenance-detail').textContent=`CSV SHA-256: ${state.catalog.provenance.source_sha256}. Cache contains original group-order scalar fields and exact source strings.`;
    mountPanes();await loadEpisodes();requestAnimationFrame(tick);
  }catch(error){$('#loading').hidden=true;showError(error);}
}
init();
