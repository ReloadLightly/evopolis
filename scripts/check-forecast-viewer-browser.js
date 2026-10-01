// Run in the actual local viewer with agent-browser eval --stdin.
// Every value below is compared with the saved Python forecast archive API.
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
  const catalog=await(await fetch('/api/catalog')).json(), cells=catalog.forecasts.cells.filter(e=>!['P0','P1','H0','H1'].includes(e.family));
  assert(cells.length===1728,'Incomplete principal forecast design');
  const groups=new Set(cells.map(e=>JSON.stringify(e.key))), conditions=new Set(cells.map(e=>[e.family,e.budget,e.training_seed].join('/')));
  assert(groups.size===24&&conditions.size===18,'Human groups or checkpoint conditions missing');
  $('#forecast-mode').click();await wait(ready);
  assert($('#compare').checked&&$('#compare').disabled,'Forecast and human must share playback');
  assert($('#forecast-group').options.length===24,'Some human groups inaccessible');
  let record=await getForecast(), game=record.forecast;
  assert(game.id===catalog.forecasts.default_id&&game.branch_index===0,'Default is not frozen branch zero');
  assert(game.family==='recurrent'&&game.budget==='continued'&&game.training_seed===17&&game.origin===5&&game.mechanism==='Mixed','Default condition differs from protocol');
  const defaultId=game.id;
  pass('complete forecast design and deterministic default',{groups:groups.size,conditions:conditions.size,cells:cells.length,id:game.id,branch_index:game.branch_index,rollout_seed:game.rollout_seed,checkpoint_sha256:game.checkpoint_sha256});
  $('[data-pane="0"] [data-resident="3"]').click();
  const exactRow=(pane,row)=>{
    const values=$$(`[data-pane="${pane}"] .exact-values tr`)[3].querySelectorAll('td');
    assert(Number(values[1].textContent)===row.offers[3],'Allocation changed in playback');
    assert(Number(values[2].textContent)===row.contributions[3],'Return changed in playback');
    assert(Number(values[4].textContent)===row.cumulative_surplus[3],'Cumulative value changed while seeking');
  };
  seek(4);exactRow(0,game.rounds[4]);exactRow(1,record.observed.rounds[4]);
  assert($$('[data-pane="0"] .source-badge')[0].textContent==='RECORDED HUMAN PREFIX','Observed prefix mislabeled');
  assert($('[data-pane="0"] .prediction-panel').hidden,'A forecast was invented for observed prefix');
  seek(5);exactRow(0,game.rounds[5]);exactRow(1,record.observed.rounds[5]);
  assert($('[data-pane="1"] .source-badge').textContent==='OBSERVED HUMAN FUTURE','Actual continuation mislabeled');
  assert($('[data-pane="0"] .forecast-status').textContent.includes('Current pool and offers are recorded'),'Precise forecast boundary missing');
  const prediction=game.rounds[5].predictions[3];
  assert(Number($('[data-pane="0"] .prediction-values dd').title)===prediction.expected_contribution,'Prediction summary differs from saved Python result');
  const axes=$$('.timeline').map(e=>({...e.dataset}));
  assert(axes[0].roundMax==='25'&&axes[0].roundMax===axes[1].roundMax,'Window clocks differ');
  assert(axes[0].poolMax===axes[1].poolMax&&axes[0].surplusMax===axes[1].surplusMax,'Axes rescaled separately');
  const band=game.bands[0];
  assert($('[data-pane="0"] .forecast-timeline').dataset.branchCount==='64'&&$('[data-pane="0"] .forecast-band').getAttribute('points').length>30,'Ensemble uncertainty band missing');
  pass('recorded prefix, precise boundary, actual future and exact resident prediction',{round:6,resident:'D',offers:game.rounds[5].offers,returns:game.rounds[5].contributions,prediction,band,axes});
  await change('#forecast-branch','median');record=await getForecast();
  assert(record.forecast.branch_selection==='median','Median illustration selection failed');
  await change('#forecast-branch','first');
  assert((await getForecast()).forecast.rollout_seed===game.rollout_seed,'Branch zero did not regenerate identically');
  pass('illustration selection preserves the 64-branch distribution',{first_branch:game.branch_index,median_branch:record.forecast.branch_index});
  for(const family of ['constant','linear','feedforward','recurrent']){
    await change('#forecast-family',family);await change('#forecast-seed','43');
    if(['constant','linear'].includes(family))assert($('#forecast-budget').options[1].disabled,'Nonexistent continued reference offered');
    else{await change('#forecast-budget','original');await change('#forecast-budget','continued');}
    for(const origin of ['0','5','10','20']){
      await change('#forecast-origin',origin);
      assert(Number($('#scrubber').max)===Number(origin)+19,'Forecast window exceeds declared horizon');
      seek(Number($('#scrubber').max));assert(Number($('#scrubber').value)<=39,'Forecast continued past recorded round 40');
    }
  }
  pass('model, budget, training seed and all origins',{families:4,origins:[0,5,10,20],maximum_playback_round:40});
  await change('#speed','4');seek(38);$('#play').click();await wait(()=>$('#play').getAttribute('aria-label')==='Play');
  assert($('#scrubber').value==='39','End playback exceeded the 40-round episode');
  $('#previous').click();assert($('#scrubber').value==='38','Previous round failed');
  $('#restart').click();assert($('#scrubber').value==='0','Restart failed');
  pass('play, previous and restart obey the episode limit',{final_round:40});
  // A failed forecast remains reachable through the same menus as every group.
  let failed=null;
  for(const cell of cells.filter(e=>e.mechanism==='Equal'&&e.family==='recurrent'&&e.training_seed===17&&e.origin===5)){
    const candidate=await(await fetch(`/api/forecast?id=${encodeURIComponent(cell.id)}&branch=first`)).json();
    if(candidate.forecast.rounds.some(r=>r.padded)){failed=candidate.forecast;break;}
  }
  assert(failed,'Expected an archived terminated Equal forecast');
  await change('#forecast-group',JSON.stringify(failed.key));await change('#forecast-family',failed.family);
  await change('#forecast-seed',String(failed.training_seed));await change('#forecast-budget',failed.budget);await change('#forecast-origin',String(failed.origin));
  seek(failed.actual_rounds);
  assert($('[data-pane="0"] .source-badge').textContent==='FORECAST · ZERO PADDING','Termination padding presented as behavior');
  assert($('[data-pane="0"] .prediction-panel').hidden,'Prediction shown for a padded decision');
  pass('failed forecast accessible and padding explicit',{id:failed.id,actual_rounds:failed.actual_rounds});
  // Return to the protocol default and a forecast round for the screenshot.
  const initial=cells.find(e=>e.id===defaultId);
  await change('#forecast-group',JSON.stringify(initial.key));await change('#forecast-family',initial.family);
  await change('#forecast-seed',String(initial.training_seed));await change('#forecast-budget',initial.budget);await change('#forecast-origin',String(initial.origin));
  seek(initial.origin+2);$('[data-pane="0"] [data-resident="3"]').click();
  assert($('#error').hidden,'Visible application error');
  window.evopolisForecastBrowserVerification={viewport:[innerWidth,innerHeight],checks};
  return window.evopolisForecastBrowserVerification;
})()
