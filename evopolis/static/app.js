import {drawTown, hitResident} from './town.js';

const $ = (selector, parent = document) => parent.querySelector(selector);
const $$ = (selector, parent = document) => [...parent.querySelectorAll(selector)];
const letters = ['A', 'B', 'C', 'D'];
const colors = {Equal:'#ff8b8b', Mixed:'#ffcf6e', Proportional:'#6dcff6', 'Recorded RL M1':'#73e0b3', Interpolating:'#b6c6df'};
const familyNames = {constant:'Constant', linear:'Linear', feedforward:'Feedforward', recurrent:'Recurrent GRU'};
const defaults = [0.75, 0.75, 0.75, 0.25];
const state = {
  mode:'recorded', compare:false, round:0, phase:0, playing:false,
  reduced:matchMedia('(prefers-reduced-motion: reduce)').matches, speed:1,
  catalog:null, generation:0, busy:true,
  panes:[0,1].map(i => ({cohort:'human', mechanism:i ? 'Proportional':'Equal',
    id:null, family:'recurrent', trainingSeed:null, selected:0, rule:i ? 'proportional':'equal', fractions:[...defaults], episode:null, element:null})),
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
function candidates(pane, anySeed=false) {
  return state.catalog.episodes.filter(e=>e.cohort===pane.cohort && e.mechanism===pane.mechanism &&
    (pane.cohort!=='trained'||e.family===pane.family&&(anySeed||e.training_seed===pane.trainingSeed)));
}
function adoptEpisode(pane, episode) {
  pane.id=episode.id;pane.cohort=episode.cohort;pane.mechanism=episode.mechanism;
  if(episode.cohort==='trained'){pane.family=episode.family;pane.trainingSeed=episode.training_seed;}
}
function medianEpisode(episodes) {
  const ordered=[...episodes].sort((a,b)=>a.surplus-b.surplus);
  const n=ordered.length, low=ordered[Math.floor((n-1)/2)].surplus, high=ordered[Math.floor(n/2)].surplus;
  const key=e=>JSON.stringify([e.condition,e.launch_id,e.episode_id]);
  const stableOrder=(a,b)=>a.cohort==='trained'
    ? a.family.localeCompare(b.family)||a.training_seed-b.training_seed||a.mechanism.localeCompare(b.mechanism)||a.rollout_index-b.rollout_index||a.id.localeCompare(b.id)
    : key(a)<key(b)?-1:key(a)>key(b)?1:0;
  // The two middle outcomes are equidistant from the exact median. Avoid a
  // floating midpoint that could accidentally override the stable key tie.
  return ordered.filter(e=>e.surplus===low||e.surplus===high).sort(stableOrder)[0];
}
function selectionOptions(pane) {
  const trained=pane.cohort==='trained', el=pane.element;
  const mechanism=$('.mechanism',el);
  const mechanisms=trained?['Equal','Mixed','Proportional','Interpolating']:['Equal','Mixed','Proportional','Recorded RL M1'];
  if(!mechanisms.includes(pane.mechanism))pane.mechanism='Equal';
  mechanism.replaceChildren(...mechanisms.map(name=>new Option(name==='Interpolating'?'Interpolating · exploratory transfer':name,name)));
  mechanism.value=pane.mechanism;
  $('.episode-field',el).hidden=trained;
  $$('.trained-field',el).forEach(field=>field.hidden=!trained);
  el.classList.toggle('trained-community',trained);
  $('.family',el).value=pane.family;
  if(trained){
    const all=candidates(pane,true);
    if(!all.some(e=>e.training_seed===pane.trainingSeed))pane.trainingSeed=17;
    const checkpoints=[...new Map(all.map(e=>[e.training_seed,e])).values()].sort((a,b)=>a.training_seed-b.training_seed);
    $('.checkpoint',el).replaceChildren(...checkpoints.map(e=>new Option(`Seed ${e.training_seed} · epoch ${e.selected_epoch} · ${e.checkpoint_sha256.slice(0,8)}`,e.training_seed)));
    $('.checkpoint',el).value=pane.trainingSeed;
  }
  const list=candidates(pane), select=$(trained?'.rollout':'.episode',el);
  if (!list.some(e=>e.id===pane.id)) pane.id=medianEpisode(list).id;
  select.replaceChildren(...[...list].sort((a,b)=>a.surplus-b.surplus || a.id.localeCompare(b.id)).map(e=>{
    const option=document.createElement('option'); option.value=e.id;
    option.textContent=`${trained?`Seed ${e.rollout_seed}`:`${e.launch_id} · ep ${e.episode_id}`} | ${e.surplus.toFixed(3)} / ${e.gini?.toFixed(3) ?? 'undefined'}`;
    return option;
  }));
  select.value=pane.id;
}
function mountPanes() {
  const container=$('#communities'); container.replaceChildren(); container.classList.toggle('compare',state.compare);
  activePanes().forEach((pane,index)=>{
    const element=$('#community-template').content.firstElementChild.cloneNode(true);
    pane.element=element; element.dataset.pane=index; element.setAttribute('aria-label',`Community ${index+1}`);
    $('.pane-name',element).textContent=`COMMUNITY ${index+1} / ${state.mode==='sandbox' ? 'SANDBOX':'ARCHIVE'}`;
    $('.recorded-selectors',element).hidden=state.mode==='sandbox';
    $('.sandbox-selectors',element).hidden=state.mode!=='sandbox';
    $('.outcome-detail',element).hidden=state.mode==='sandbox';
    $('.cohort',element).value=pane.cohort; $('.mechanism',element).value=pane.mechanism;
    $('.cohort option[value="trained"]',element).disabled=!state.catalog.trained_default_id;
    selectionOptions(pane);
    $('.cohort',element).addEventListener('change',e=>{pane.cohort=e.target.value;pane.id=null;if(pane.cohort==='trained')adoptEpisode(pane,state.catalog.episodes.find(e=>e.id===state.catalog.trained_default_id));selectionOptions(pane);updateContext();loadEpisodes(true);});
    $('.mechanism',element).addEventListener('change',e=>{pane.mechanism=e.target.value;pane.id=null;selectionOptions(pane);loadEpisodes(true);});
    $('.episode',element).addEventListener('change',e=>{pane.id=e.target.value;loadEpisodes(true);});
    $('.family',element).addEventListener('change',e=>{pane.family=e.target.value;pane.id=null;selectionOptions(pane);loadEpisodes(true);});
    $('.checkpoint',element).addEventListener('change',e=>{pane.trainingSeed=Number(e.target.value);pane.id=null;selectionOptions(pane);loadEpisodes(true);});
    $('.rollout',element).addEventListener('change',e=>{pane.id=e.target.value;loadEpisodes(true);});
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
  updateContext();
}
function updateContext(){
  $('#mode-context').textContent=state.mode==='sandbox'
    ? (state.compare ? 'SCRIPTED COMPARISON · Same fractions copied on entry; each pane can be explicitly edited. Outcomes depend on these fixed assumptions.' : 'NEW SCRIPTED SIMULATION · Fixed-fraction returns using the published equation. No human prediction, fitting or learning.')
    : state.compare ? 'SYNCHRONIZED COMPARISON · Common round coordinates and shared axes. Different observed groups and simulations are not individual counterfactuals.'
    : activePanes().some(p=>p.cohort==='trained') ? 'NEW TRAINED EVOPOLIS AGENTS · 3,072 generated games from all 12 best-validation checkpoints. Frozen weights; separate resident histories.'
    : 'RECORDED EVIDENCE · 160 human communities + 2,048 upstream BC1 games. Every eligible episode is available.';
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
    const episodes=await Promise.all(panes.map(p=>state.mode!=='sandbox'
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
  drawTown($('.town',pane.element),{round,phase:finished||round.padded?2:state.phase,selectedResident:pane.selected,reducedMotion:state.reduced||round.padded,progress});
}
function cumulativeGini(values) {
  const total=values.reduce((a,b)=>a+b,0);
  return total===0 ? null : values.flatMap(a=>values.map(b=>Math.abs(a-b))).reduce((a,b)=>a+b,0)/(8*total);
}
function renderPane(pane) {
  const el=pane.element, episode=pane.episode;if(!episode)return;
  const selectedIndex=Math.min(state.round,episode.rounds.length-1), r=episode.rounds[selectedIndex], i=pane.selected;
  const isScript=episode.cohort==='scripted', isTrained=episode.cohort==='trained', isSimulated=isScript||isTrained, finished=state.round>=episode.rounds.length;
  const source=$('.source-badge',el);source.className=`source-badge ${episode.cohort}`;
  source.textContent=isScript?'NEW SCRIPTED SIMULATION':isTrained?'NEW TRAINED EVOPOLIS SIMULATION':episode.cohort==='human'?'RECORDED HUMAN · EXP 1':'RECORDED UPSTREAM MODEL · BC1';
  $('.identity',el).textContent=isScript
    ? `${episode.mechanism} · q = [${pane.fractions.join(', ')}] · ${episode.rounds.length} rounds · ${episode.termination}`
    : isTrained ? `${familyNames[episode.family]} · training seed ${episode.training_seed} · selected epoch ${episode.selected_epoch} · checkpoint ${episode.checkpoint_sha256.slice(0,12)} · rollout seed ${episode.rollout_seed} · ${episode.mechanism}${episode.mechanism==='Interpolating'?' (exploratory transfer)':''}`
    : `${episode.condition} · launch ${episode.launch_id} · episode ${episode.episode_id} · source rounds 0–39`;
  $('.identity',el).title=isTrained?`Checkpoint SHA-256: ${episode.checkpoint_sha256}`:'';
  $$('.phase-strip span',el).forEach((span,index)=>span.classList.toggle('active',index===(finished||r.padded?2:state.phase)));
  $('.pool-before',el).textContent=fmt(r.pool_before);$('.pool-before',el).title=exact(r.pool_before);
  $('.pool-after',el).textContent=fmt(r.pool_after);$('.pool-after',el).title=r.pool_after==null?'No following observation':exact(r.pool_after);
  $('.after-label',el).textContent=isSimulated?'Simulated pool after':'Recorded pool after';
  $('.active-count',el).textContent=`${r.active_count} / 4`;
  $('.total-surplus',el).textContent=fmt(r.group_cumulative_surplus ?? r.cumulative_surplus.reduce((a,b)=>a+b,0));
  $('.resident-name',el).textContent=`Resident ${letters[i]}`;
  $('.resident-position',el).textContent=`Position ${i} in this group · values in resource units`;
  $$('.resident-buttons button',el).forEach((b,j)=>b.setAttribute('aria-pressed',j===i?'true':'false'));
  $('.resident-values',el).innerHTML=[['Allocation',r.offers[i]],['Returned',r.contributions[i]],['Retained this round',r.surplus[i]],['Cumulative retained',r.cumulative_surplus[i]]].map(([label,value])=>`<div><dt>${label}</dt><dd title="${esc(exact(value))}">${fmt(value)}</dd></div>`).join('');
  const opportunity=$('.opportunity',el);opportunity.classList.toggle('low',r.offers[i]<1);
  opportunity.textContent=r.padded?`Post-termination padding. No decision was sampled; the game ended after round ${episode.actual_rounds}.`:r.offers[i]<1?'Offer below 1 unit in this round. This alone does not establish permanent exclusion.':'Offer at least 1 unit in this round.';
  $('.history',el).innerHTML=episode.rounds.slice(0,selectedIndex+1).map(h=>`<tr><td>${h.round_id+1}${h.padded?' · padded':''}</td>${[h.offers[i],h.contributions[i],h.surplus[i],h.cumulative_surplus[i]].map(v=>`<td title="${esc(exact(v))}">${fmt(v)}</td>`).join('')}</tr>`).join('');
  const previous=selectedIndex ? episode.rounds[selectedIndex-1] : null;
  $('.information-basis',el).textContent=isScript
    ? 'This fixed script uses only its own allocation and chosen fraction q. Public history below is researcher context, not a script input.'
    :isTrained
      ? `All families receive the same nine prepared observations: current offers, previous returns and current pool, rotated self-first and divided by 200. ${episode.family==='recurrent'?'This GRU also uses its own recurrent state, reset at the start of the game.':episode.family==='constant'?'The constant model ignores input values except the legal action support.':'This family has no recurrent state; previous returns already provide one-step history.'} Weights remain frozen. Cumulative totals and the playback horizon are researcher context, not model inputs.`
    :episode.cohort==='bc1'
      ? 'The upstream BC1 model used nine inputs: all current offers, previous returns and current pool, plus recurrent memory. Cumulative totals below are researcher context, not a separate model input.'
      : 'People saw all current offers, previous public returns and pool size; earlier public outcomes and cumulative surplus could be remembered. Current returns were simultaneous.';
  $('.previous-returns',el).textContent=`Previous returns A–D: ${(previous?.contributions||[0,0,0,0]).map(exact).join(' · ')}`;
  $('.prior-cumulative',el).textContent=`Public retained totals before this round A–D: ${(previous?.cumulative_surplus||[0,0,0,0]).map(exact).join(' · ')}`;
  const prediction=isTrained&&!r.padded?r.predictions?.[i]:null;
  $('.prediction-panel',el).hidden=!prediction;
  $('.prediction-note',el).textContent=isTrained
    ? r.padded?'No prediction or memory update was run in these padded rounds.':prediction?'Predictions are saved before sampling any current-round return. Four conditionally independent draws use the shared pre-decision state.':'No saved prediction summary is available for this round.'
    : 'No fitted model predictions are available for this record.';
  if(prediction){
    $('.prediction-values',el).innerHTML=[['Expected return',prediction.expected_contribution],['P(return = 0)',prediction.p_zero],['P(return = maximum)',prediction.p_max]].map(([label,value])=>`<div><dt>${label}</dt><dd title="${esc(exact(value))}">${fmt(value)}</dd></div>`).join('');
    $('.prediction-support',el).textContent=`Legal integers: 0–${prediction.legal_max}. Probabilities are 0–1; endpoints coincide when the maximum is zero. These are model predictions, not recorded human beliefs.`;
  }
  const raw=r.raw||{};
  const rawVector=(name,fallback)=>fallback.map((value,j)=>raw[`${name}_${j}`] ?? exact(value));
  const offers=rawVector('offer',r.offers),returns=rawVector('player_action',r.contributions),surplus=rawVector('player_reward',r.surplus);
  $('.exact-values',el).innerHTML=letters.map((letter,j)=>`<tr><td>${letter}</td><td>${esc(offers[j])}</td><td>${esc(returns[j])}</td><td>${esc(surplus[j])}</td><td>${esc(exact(r.cumulative_surplus[j]))}</td></tr>`).join('');
  $('.exact-pool',el).textContent=`Pool before: ${raw['mechanism_observation.pool'] ?? exact(r.pool_before)} · ${isSimulated?'Simulated':'Recorded'} after: ${r.pool_after_raw ?? exact(r.pool_after)}`;
  $('.accounting',el).textContent=`Equation estimate: ${exact(r.equation_after)} · ${isSimulated?'Simulated':'Recorded'} − equation: ${r.pool_after==null?'Unavailable':exact(r.pool_after-r.equation_after)}. ${isSimulated?'The Python numerical environment has no undocumented 0.01 floor.':'Observations are never replaced by this estimate.'}`;
  $('.after-source',el).textContent=`Next-pool source: ${r.after_source}. ${isSimulated?'Internal':'Source'} round ${r.round_id}; playback round ${r.round_id+1}.${isTrained?` Checkpoint SHA-256: ${episode.checkpoint_sha256}. Generated mean surplus divides full-game retained resources by 4 × 40, including zero padding.`:''}`;
  const gini=cumulativeGini(r.cumulative_surplus);
  $('.gini-note',el).textContent=`Cumulative retained sums ${isSimulated?'simulated':'recorded'} round surplus through this round${isSimulated?'':' (not the rounded source cumulative counter)'}. Current-game cumulative Gini: ${gini==null?'Undefined (zero total)':exact(gini)}. Four-player maximum 0.75. Completed-game Gini: ${episode.gini==null?'Undefined (zero total)':exact(episode.gini)}.${raw.players_cumulative_reward ? ` Source cumulative counter A–D: ${raw.players_cumulative_reward}`:''}`;
  $('.end-state',el).textContent=finished?`Run ended after round ${episode.rounds.length} (${episode.termination}); final state held while the other pane continues.`
    :isTrained ? `${r.padded?'POST-TERMINATION PADDING · ':''}${episode.actual_rounds} executed rounds; ${episode.termination}. ${episode.actual_rounds<40?'The remaining rounds have zero resources and returns; cumulative retained stays fixed.':'Full 40-round horizon.'}`
    :r.pool_after==null?'Final BC1 next pool is unavailable; no later observation was released.'
    :isScript&&selectedIndex===episode.rounds.length-1?`Run ends: ${episode.termination}. No undocumented 0.01 floor.`
    :`Units: flowers / retained resources. All four contributions were simultaneous.`;
  drawTimeline(pane,selectedIndex);
  drawScene(pane);
  if(!isScript)drawScatter(pane);
}
function drawTimeline(pane,index) {
  const rows=pane.episode.rounds, horizon=40, x=j=>38+j*470/(horizon-1);
  const groupTotal=r=>r.group_cumulative_surplus ?? r.cumulative_surplus.reduce((a,b)=>a+b,0);
  const cumulative=rows.map(groupTotal);
  // Both panes use one domain for each quantity, including a short sandbox.
  const maxSurplus=Math.max(1,...activePanes().flatMap(p=>p.episode.rounds.map(groupTotal)));
  const yPool=v=>100-v/200*78,ySurplus=v=>100-v/maxSurplus*78;
  const path=(values,y)=>values.map((v,j)=>`${j?'L':'M'}${x(j)},${y(v)}`).join(' ');
  const svg=$('.timeline',pane.element), ending=pane.episode.actual_rounds;
  svg.dataset.roundMax=horizon;svg.dataset.poolMax=200;svg.dataset.surplusMax=maxSurplus;
  const marker=ending&&ending<40?`<rect x="${x(ending)}" y="22" width="${508-x(ending)}" height="78" fill="#3b5278" opacity=".25"/><path d="M${x(ending-1)} 18V104" stroke="#ff8b8b" stroke-dasharray="2 3"/>${svgText(Math.min(470,x(ending-1)+4),12,`end ${ending}`,'#ff8b8b')}`:'';
  svg.innerHTML=`<title>Common rounds 1–40. Pool before each round (left axis 0–200); total cumulative retained (shared right axis 0–${maxSurplus}). Cursor at round ${index+1}.${ending&&ending<40?` Game ended after round ${ending}; later rounds are padding.`:''}</title><path d="M38 22V100H508" fill="none" stroke="#3b5278"/><path d="M38 61H508 M38 22H508" fill="none" stroke="#253a5e"/>${marker}${svgText(31,26,'200','#6dcff6','end')}${svgText(31,104,'0','#6dcff6','end')}${svgText(515,26,fmt(maxSurplus),'#ffcf6e')}${svgText(515,104,'0','#ffcf6e')}${svgText(38,120,'1')}${svgText(270,120,'Round · shared axes','#b6c6df','middle')}${svgText(508,120,horizon,'#b6c6df','end')}<path d="${path(rows.map(r=>r.pool_before),yPool)}" fill="none" stroke="#6dcff6" stroke-width="2"/><path d="${path(cumulative,ySurplus)}" fill="none" stroke="#ffcf6e" stroke-width="2"/><path d="M${x(state.round)} 15V104" stroke="#fff1d2" stroke-dasharray="3 3"/><circle cx="${x(index)}" cy="${yPool(rows[index].pool_before)}" r="3" fill="#6dcff6"/><circle cx="${x(index)}" cy="${ySurplus(cumulative[index])}" r="3" fill="#ffcf6e"/>`;
}
function drawScatter(pane) {
  const list=candidates(pane),svg=$('.scatter',pane.element), color=colors[pane.mechanism];
  const maxSurplus=Math.max(14,...activePanes().flatMap(p=>candidates(p).map(e=>e.surplus)));
  const undefinedCount=list.filter(e=>e.gini==null).length;
  $('.selection-rule',pane.element).textContent=pane.cohort==='trained'
    ? `Initial trained display: median-surplus GRU/Equal game at predeclared training seed 17, with ties broken by rollout index. It does not select a superior checkpoint. Other selections use the same median rule within the selected family, seed and mechanism. Menus expose all checkpoints and 64 rollout seeds per condition. This plot shows the selected checkpoint's games; ${undefinedCount} undefined Gini values remain available in the menu. Mean surplus uses 4 × 40 as denominator.`
    : 'Default within each condition: episode nearest the median mean surplus, with ties broken by the full episode key. All games are shown; selection is not a representative causal comparison.';
  svg.innerHTML=`<path d="M40 15V155H520" fill="none" stroke="#3b5278"/>${svgText(35,20,fmt(maxSurplus),'#b6c6df','end')}${svgText(35,158,'0','#b6c6df','end')}${svgText(40,170,'0')}${svgText(520,170,'0.75','#b6c6df','end')}${svgText(285,185,'Completed-game Gini','#b6c6df','middle')}${svgText(45,12,'Mean surplus / player / round')}`;
  for(const e of list){
    if(e.gini==null)continue;
    const x=40+e.gini/0.75*480,y=155-e.surplus/maxSurplus*130,r=e.id===pane.id?6:3;
    const mark=document.createElementNS('http://www.w3.org/2000/svg',pane.mechanism==='Equal'?'circle':'path');
    if(pane.mechanism==='Equal') {mark.setAttribute('cx',x);mark.setAttribute('cy',y);mark.setAttribute('r',r);}
    else {
      const paths={Mixed:`M${x-r} ${y-r}h${2*r}v${2*r}h${-2*r}Z`,Proportional:`M${x} ${y-r*1.2}L${x+r} ${y+r}H${x-r}Z`,'Recorded RL M1':`M${x} ${y-r*1.3}L${x+r} ${y}L${x} ${y+r*1.3}L${x-r} ${y}Z`,Interpolating:`M${x-r} ${y-r}L${x+r} ${y+r}M${x-r} ${y+r}L${x+r} ${y-r}`};
      mark.setAttribute('d',paths[pane.mechanism]);
    }
    mark.setAttribute('fill',color);mark.setAttribute('opacity',e.id===pane.id?1:0.65);mark.setAttribute('stroke',e.id===pane.id?'#fff1d2':pane.mechanism==='Interpolating'?color:'none');
    mark.setAttribute('tabindex','0');mark.setAttribute('role','button');
    mark.setAttribute('aria-label',`${pane.cohort==='trained'?`Rollout seed ${e.rollout_seed}`:`Episode ${e.launch_id}, ${e.episode_id}`}; surplus ${e.surplus.toFixed(3)}, Gini ${e.gini.toFixed(3)}`);
    const choose=()=>{pane.id=e.id;$(pane.cohort==='trained'?'.rollout':'.episode',pane.element).value=e.id;loadEpisodes(true);};
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
    if(state.mode!=='sandbox')params.set(`episode${i}`,p.id);
    else {params.set(`rule${i}`,p.rule);params.set(`q${i}`,p.fractions.join(','));}
  });
  history.replaceState(null,'',`?${params}`);
}
function setMode(mode) {
  if(mode===state.mode&&(mode==='sandbox'||activePanes().every(p=>(p.cohort==='trained')===(mode==='trained'))))return;pause();
  if(mode==='trained'){
    state.panes.forEach(p=>{if(p.cohort!=='trained'){p.recordedId=p.id;adoptEpisode(p,state.catalog.episodes.find(e=>e.id===(p.trainedId||state.catalog.trained_default_id)));}});
  }else if(mode==='recorded'){
    state.panes.forEach(p=>{if(p.cohort==='trained'){p.trainedId=p.id;const saved=state.catalog.episodes.find(e=>e.id===p.recordedId);if(saved)adoptEpisode(p,saved);else {p.cohort='human';p.id=null;if(p.mechanism==='Interpolating')p.mechanism='Equal';}}});
  }
  state.mode=mode;
  if(mode==='sandbox'&&state.compare)state.panes[1].fractions=[...state.panes[0].fractions];
  updateModeButtons();
  mountPanes();loadEpisodes(true);
}
function updateModeButtons(){for(const mode of ['recorded','trained','sandbox']){const button=$(`#${mode}-mode`);button.classList.toggle('active',state.mode===mode);button.setAttribute('aria-pressed',state.mode===mode);}}
$('#recorded-mode').addEventListener('click',()=>setMode('recorded'));
$('#trained-mode').addEventListener('click',()=>setMode('trained'));
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
    const initialMode=params.get('mode')==='sandbox'?'sandbox':params.get('mode')==='trained'&&state.catalog.trained_default_id?'trained':'recorded';
    $('#trained-mode').disabled=!state.catalog.trained_default_id;
    if(initialMode==='trained')state.panes.forEach(p=>adoptEpisode(p,state.catalog.episodes.find(e=>e.id===state.catalog.trained_default_id)));
    state.panes.forEach((pane,i)=>{
      const saved=state.catalog.episodes.find(e=>e.id===params.get(`episode${i}`));
      if(saved)adoptEpisode(pane,saved);
      const resident=Number(params.get(`resident${i}`));if(Number.isInteger(resident)&&resident>=0&&resident<4)pane.selected=resident;
      const rule=params.get(`rule${i}`);if(['equal','mixed','proportional','interpolating'].includes(rule))pane.rule=rule;
      const fractions=params.get(`q${i}`)?.split(',').map(Number);if(fractions?.length===4&&fractions.every(v=>Number.isFinite(v)&&v>=0&&v<=1))pane.fractions=fractions;
    });
    const round=Number(params.get('round'));if(Number.isInteger(round)&&round>=1&&round<=40)state.round=round-1;
    state.mode=initialMode;
    updateModeButtons();
    $('#provenance-detail').textContent=`CSV SHA-256: ${state.catalog.provenance.source_sha256}. Recorded cache contains original group-order scalar fields and exact source strings. ${state.catalog.trained_default_id?'The trained archive preserves checkpoint identities and decimal-string rollout seeds; no fresh model computation occurs during playback.':''}`;
    mountPanes();await loadEpisodes();requestAnimationFrame(tick);
  }catch(error){$('#loading').hidden=true;showError(error);}
}
init();
