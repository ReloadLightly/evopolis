// Run in the real, freshly opened viewer using agent-browser eval --stdin.
// Python verifies checkpoint reloading/regeneration separately. These checks
// follow controls and compare displayed values with the published archive API.
(async () => {
  const checks=[], $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
  const assert=(condition,message)=>{if(!condition)throw new Error(message);};
  const wait=async predicate=>{const end=Date.now()+15000;while(!predicate()){if(Date.now()>end)throw new Error('UI did not become ready');await new Promise(r=>setTimeout(r,70));}};
  const ready=()=>$('#loading').hidden&&$('#error').hidden&&!$('#play').disabled;
  const change=async(selector,value)=>{const el=$(selector);el.value=value;el.dispatchEvent(new Event('change',{bubbles:true}));await wait(ready);};
  const seek=value=>{const el=$('#scrubber');el.value=value;el.dispatchEvent(new Event('input',{bubbles:true}));};
  const pass=(name,evidence)=>checks.push({name,status:'pass',evidence});
  await wait(ready);
  const catalog=await(await fetch('/api/catalog')).json(), trained=catalog.episodes.filter(e=>e.cohort==='trained');
  assert(trained.length===3072,'Missing generated games');
  const counts=new Map();for(const e of trained){const key=[e.family,e.training_seed,e.mechanism].join('/');counts.set(key,(counts.get(key)||0)+1);assert(typeof e.rollout_seed==='string','Rollout seed lost integer precision');}
  assert(counts.size===48&&[...counts.values()].every(n=>n===64),'Incomplete family/seed/mechanism design');
  if($('#compare').checked){$('#compare').click();await wait(ready);}
  $('#trained-mode').click();await wait(ready);
  const defaultId=$('.rollout').value;
  assert(defaultId===catalog.trained_default_id,'Wrong trained display default');
  const game=await(await fetch(`/api/episode?id=${encodeURIComponent(defaultId)}`)).json();
  assert(game.family==='recurrent'&&game.training_seed===17&&game.mechanism==='Equal','Default must use predeclared GRU seed 17 / Equal');
  assert($('.checkpoint').options.length===3&&$('.rollout').options.length===64,'Checkpoint/rollout options missing');
  assert($('.source-badge').textContent==='NEW TRAINED EVOPOLIS SIMULATION','Trained provenance missing');
  pass('complete design and default',{games:trained.length,cells:counts.size,id:defaultId,rollout_seed:game.rollout_seed,checkpoint_sha256:game.checkpoint_sha256});
  $('[data-resident="3"]').click();assert($('.resident-name').textContent==='Resident D','Resident inspection failed');
  const verifyRound=index=>{
    const row=game.rounds[index],cells=$$('.exact-values tr')[3].querySelectorAll('td');
    assert(Number(cells[1].textContent)===row.offers[3],'Displayed allocation differs from generated Python output');
    assert(Number(cells[2].textContent)===row.contributions[3],'Displayed return differs from generated Python output');
    assert(Number(cells[4].textContent)===row.cumulative_surplus[3],'Cumulative seeking drift');
  };
  seek(39);verifyRound(39);seek(0);verifyRound(0);
  const prediction=game.rounds[0].predictions[3];
  assert(Number($('.prediction-values dd').title)===prediction.expected_contribution,'Before-choice prediction differs from saved checkpoint prediction');
  pass('resident, exact values, prediction and backward seek',{resident:'D',first_prediction:prediction,final_total:game.rounds[39].cumulative_surplus[3]});
  await change('#speed','4');$('#play').click();await wait(()=>Number($('#scrubber').value)>=1);$('#play').click();
  const paused=$('#scrubber').value;await new Promise(r=>setTimeout(r,450));assert($('#scrubber').value===paused,'Pause failed');
  $('#previous').click();assert(Number($('#scrubber').value)===Number(paused)-1,'Previous round failed');
  $('#restart').click();assert($('#scrubber').value==='0','Restart failed');
  pass('play, pause, previous, restart',{paused_round:Number(paused)+1});
  for(const family of ['constant','linear','feedforward','recurrent']){
    await change('.family',family);assert($('.checkpoint').options.length===3,'Missing training seed');
    await change('.checkpoint','43');assert($('.rollout').options.length===64,'Missing rollout seeds');
    seek(5);await change('.mechanism','Interpolating');assert($('#scrubber').value==='0','Mechanism change did not reset');
    assert($('.identity').textContent.includes('exploratory transfer'),'Interpolating transfer label missing');
    const first=$('.rollout').options[0].value,last=$('.rollout').options[63].value;
    await change('.rollout',last);await change('.rollout',first);
  }
  pass('family, checkpoint, mechanism and rollout selection',{families:4,training_seed:43,rollouts_per_cell:64});
  const ended=trained.find(e=>e.actual_rounds<40);
  assert(ended,'Expected an exact-zero generated game to inspect padding');
  await change('.family',ended.family);await change('.checkpoint',String(ended.training_seed));await change('.mechanism',ended.mechanism);await change('.rollout',ended.id);
  seek(ended.actual_rounds);assert($('.end-state').textContent.includes('POST-TERMINATION PADDING'),'Padding presented as a new decision');
  assert($('.prediction-panel').hidden,'Padding invented a sampled prediction');
  assert($('.timeline').textContent.includes(`Game ended after round ${ended.actual_rounds}`),'Actual ending missing on timeline');
  pass('exact-zero termination and padding',{id:ended.id,executed_rounds:ended.actual_rounds});
  $('#compare').click();await wait(ready);
  await change('[data-pane="1"] .cohort','human');seek(8);
  const axes=$$('.timeline').map(e=>({...e.dataset}));
  assert(axes[0].roundMax==='40'&&axes[0].roundMax===axes[1].roundMax,'Round coordinates differ');
  assert(axes[0].poolMax===axes[1].poolMax&&axes[0].surplusMax===axes[1].surplusMax,'Comparison quantities rescaled separately');
  assert($$('.after-source').every(e=>e.textContent.includes('round 8; playback round 9')),'Comparison clocks differ');
  assert($$('[data-pane="1"] .episode option').length===40,'Human replay lost');
  pass('independent human/trained comparison with common scales',{axes,sources:$$('.source-badge').map(e=>e.textContent)});
  const again=await(await fetch(`/api/episode?id=${encodeURIComponent(defaultId)}`)).json();
  assert(JSON.stringify(again)===JSON.stringify(game),'Repeated archive request changed trajectory');
  assert(!document.body.textContent.includes('undefinedundefined'),'Broken metadata');
  assert($('#error').hidden,'Visible application error');
  pass('published episode identity is reproducible',{id:defaultId,rollout_seed:again.rollout_seed});
  window.evopolisTrainedBrowserVerification={viewport:[innerWidth,innerHeight],checks};
  return window.evopolisTrainedBrowserVerification;
})()
