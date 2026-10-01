// Run in the local observatory using agent-browser eval --stdin.
// Inspect the real saved Task 05 archive through the same API as the viewer.
(async () => {
  const checks=[], $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
  const assert=(condition,message)=>{if(!condition)throw new Error(message);};
  const wait=async predicate=>{const end=Date.now()+20000;while(!predicate()){if(Date.now()>end)throw new Error('UI did not become ready');await new Promise(r=>setTimeout(r,60));}};
  const ready=()=>$('#loading').hidden&&$('#error').hidden&&!$('#play').disabled;
  const change=async(selector,value)=>{const element=$(selector);element.value=value;element.dispatchEvent(new Event('change',{bubbles:true}));await wait(ready);};
  const seek=value=>{const element=$('#scrubber');element.value=value;element.dispatchEvent(new Event('input',{bubbles:true}));};
  const pass=(name,evidence)=>checks.push({name,status:'pass',evidence});
  const getForecast=async()=>{const params=new URLSearchParams(location.search);return (await fetch(`/api/forecast?id=${encodeURIComponent(params.get('forecast'))}&branch=${params.get('branch')}`)).json();};
  await wait(ready);
  const catalog=await(await fetch('/api/catalog')).json();
  const cells=catalog.forecasts.cells.filter(cell=>['P0','P1','H0','H1'].includes(cell.family));
  assert(cells.length===1152,'Task 05 principal conditions are incomplete');
  assert(new Set(cells.map(cell=>JSON.stringify(cell.key))).size===24,'Task 05 human groups are missing');
  assert(new Set(cells.map(cell=>`${cell.family}/${cell.training_seed}`)).size===12,'Task 05 fitted checkpoints are missing');
  $('#forecast-mode').click();await wait(ready);
  for(const family of ['P0','P1','H0','H1']){
    await change('#forecast-family',family);
    assert($('#forecast-budget').value==='conditional','Task 05 budget mislabeled');
    assert($('#forecast-budget').options[0].disabled&&$('#forecast-budget').options[1].disabled,'Nonexistent Task 05 budget selectable');
    for(const seed of ['17','29','43']){await change('#forecast-seed',seed);assert((await getForecast()).forecast.training_seed===Number(seed),'Wrong training seed');}
    for(const origin of ['0','5','10','20']){await change('#forecast-origin',origin);assert(Number($('#scrubber').max)===Number(origin)+19,'Wrong forecast window');}
  }
  pass('four matched families, twelve fits and all forecast origins',{families:['P0','P1','H0','H1'],seeds:[17,29,43],origins:[0,5,10,20],conditions:cells.length});
  await change('#forecast-family','H1');await change('#forecast-seed','17');await change('#forecast-origin','5');await change('#forecast-branch','first');
  let record=await getForecast(), game=record.forecast;
  seek(4);
  assert($('[data-pane="0"] .prediction-panel').hidden&&$('[data-pane="0"] .conditional-panel').hidden,'Generated predictions leaked into the recorded prefix');
  seek(5);
  for(let resident=0;resident<4;resident++){
    $(`[data-pane="0"] [data-resident="${resident}"]`).click();
    const prediction=game.rounds[5].predictions[resident];
    const values=$$('[data-pane="0"] .conditional-values dd');
    assert(Number(values[0].title)===prediction.history.own_trace,'Own trace changed in playback');
    assert(Number(values[1].title)===prediction.history.peer_trace,'Peer trace changed in playback');
    assert(!$('[data-pane="0"] .pmf-detail').hidden,'Saved legal PMF is inaccessible');
    const probabilities=$$('[data-pane="0"] .pmf-values tr').map(row=>Number(row.children[1].textContent));
    assert(JSON.stringify(probabilities)===JSON.stringify(prediction.pmf),'Saved PMF changed in playback');
    assert(Math.abs(probabilities.reduce((a,b)=>a+b,0)-1)<1e-8,'Displayed PMF is not normalized');
    assert(Number($$('[data-pane="0"] .latent-values dd')[3].title)===game.latent_effects[resident],'Wrong sampled resident effect');
  }
  assert($('[data-pane="0"] .prediction-note').textContent.includes('fixed resident effect'),'Branch conditioning scope is absent');
  assert($('[data-pane="0"] .latent-note').textContent.includes('holds it fixed'),'Latent persistence is not explained');
  const originalEffect=$$('[data-pane="0"] .latent-values dd')[3].title;
  seek(6);assert($$('[data-pane="0"] .latent-values dd')[3].title===originalEffect,'The displayed resident effect changed between rounds');
  assert($('[data-pane="1"] .prediction-panel').hidden,'Invented model predictions on actual human continuation');
  pass('exact PMFs, pre-choice histories, past-only posterior and persistent resident effects',{id:game.id,parameters:game.parameters,latent_effects:game.latent_effects,prefix_posteriors:game.prefix_posteriors});
  await change('#forecast-branch','median');record=await getForecast();
  assert(record.forecast.branch_selection==='median','Median branch selection failed');
  seek(5);assert(Number($$('[data-pane="0"] .latent-values dd')[3].title)===record.forecast.latent_effects[3],'Median branch retained branch-zero effect');
  await change('#forecast-family','P0');seek(5);
  assert($('[data-pane="0"] .latent-note').textContent.includes('fixes u and σ to zero'),'Population-only scope missing');
  pass('branch-specific effect selection and population-only family',{median_branch:record.forecast.branch_index});
  let failed=null;
  for(const cell of cells.filter(e=>e.family==='H1'&&e.training_seed===17&&e.mechanism==='Equal'&&e.origin===5)){
    const candidate=await(await fetch(`/api/forecast?id=${encodeURIComponent(cell.id)}&branch=first`)).json();
    if(candidate.forecast.rounds.some(round=>round.padded)){failed=candidate.forecast;break;}
  }
  assert(failed,'No terminated conditional-model forecast found for padding inspection');
  await change('#forecast-family','H1');await change('#forecast-group',JSON.stringify(failed.key));await change('#forecast-branch','first');seek(failed.actual_rounds);
  assert($('[data-pane="0"] .source-badge').textContent==='FORECAST · ZERO PADDING','Conditional-model termination padding is mislabeled');
  assert($('[data-pane="0"] .prediction-panel').hidden&&$('[data-pane="0"] .conditional-panel').hidden,'A padded round is presented as a model choice');
  pass('terminated conditional forecast remains accessible with explicit padding',{id:failed.id,actual_rounds:failed.actual_rounds});
  await change('#forecast-group',JSON.stringify(game.key));
  await change('#forecast-family','recurrent');
  assert($('#forecast-budget').value==='continued','Return to Task 04 continued model failed');
  seek(5);assert($('[data-pane="0"] .conditional-panel').hidden,'Task 05 histories appeared on an old GRU forecast');
  $('#recorded-mode').click();await wait(ready);
  assert($('[data-pane="0"] .source-badge').textContent.includes('RECORDED'),'Recorded replay was not preserved');
  pass('Task 04 model and recorded replay remain reachable',{old_family:'recurrent',budget:'continued'});
  $('#forecast-mode').click();await wait(ready);await change('#forecast-family','H1');await change('#forecast-branch','first');seek(5);
  $('[data-pane="0"] [data-resident="3"]').click();
  $('[data-pane="0"] .pmf-detail').open=true;
  $('[data-pane="0"] .conditional-panel details').open=true;
  assert($('#error').hidden,'Visible application error');
  window.evopolisConditionalBrowserVerification={viewport:[innerWidth,innerHeight],checks};
  return window.evopolisConditionalBrowserVerification;
})()
