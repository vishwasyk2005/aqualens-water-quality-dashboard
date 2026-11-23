import {escapeHtml as esc, formatNumber as number, formatDate, icon, download, safeUrl, toCsv, parameterNames} from './utils.js';

const indicators = ['ndwi', 'turbidity_proxy', 'chlorophyll_proxy'];
const benchmarkHorizons = [8, 16, 32];
const forecastHorizons = [...benchmarkHorizons, 60, 90, 180, 365, 730];
const shortNames = {ndwi:'Water index', turbidity_proxy:'Turbidity', chlorophyll_proxy:'Chlorophyll'};
const dayMs = 86400000;
const shortDate = value => formatDate(value, true);
const datePart = value => String(value ?? '').slice(0,10);
const timestamp = value => Date.parse(datePart(value));
const isNumber = value => value != null && value !== '' && Number.isFinite(+value);
const addDays = (value, days) => new Date(timestamp(value) + days * dayMs).toISOString().slice(0,10);
const url = (name, entries) => `/api/${name}?${new URLSearchParams(Object.entries(entries).filter(([,value])=>value!=null && value!=='')).toString()}`;
const niceModel = model => ({det:'DET', auto:'Auto', consensus:'DET + seasonal + ridge', persistence:'Last observation', recent_median:'Recent median', climatology:'Seasonal regression', seasonal_kernel:'Seasonal similarity', ridge:'Lagged ridge regression', prophet:'Prophet'})[model] || model;
const el = (host, selector) => host.querySelector(selector);

function zones(data) {
  const found = new Map();
  for (const point of data.zones || []) {
    if (point.zone_id != null) found.set(String(point.zone_id), point.zone_name || point.zone_id);
  }
  if (!found.size) {
    for (const point of data.observations || []) {
      if (point.zone_id != null && !found.has(String(point.zone_id))) found.set(String(point.zone_id), point.zone_name || point.zone_id);
    }
  }
  return [['all','Whole lake'],...found];
}

function models(data, mode, isHistorical) {
  const rows = data.model_benchmark?.models || [];
  const named = new Map(rows.map(item=>[item.id,item.name]));
  const list = new Map();
  const verified = new Map((data.models || []).map(item=>[item.id,item]));
  const selected = data.model_benchmark?.selected_model_name || verified.get('auto')?.name;
  if (mode==='snapshot') {
    const ids = Object.keys(isHistorical ? data.backtests || {} : data.forecasts || {});
    for (const model of ids) list.set(model, model==='auto' ? `Auto · ${selected || 'recommended'}` : named.get(model) || niceModel(model));
    if (isHistorical && data.backtest?.available && !list.has('det')) list.set('det','DET');
  } else if (isHistorical) {
    if (data.model_benchmark?.saved_history) {
      list.set('auto',`Auto · ${selected || 'recommended'}`);
      for(const item of rows) list.set(item.id,item.name || niceModel(item.id));
    }
    if (!list.has('det')) list.set('det','DET');
  } else {
    if (verified.get('auto')?.available) list.set('auto',`Auto · ${selected || 'recommended'}`);
    for (const item of rows) {
      if (['det','consensus'].includes(item.id) && !verified.get('det')?.available) continue;
      if (item.id==='prophet' && verified.get('prophet')?.available === false) continue;
      list.set(item.id,item.name || niceModel(item.id));
    }
    if (verified.get('det')?.available && !list.has('det')) list.set('det','DET');
  }
  return [...list];
}

function options(items, selected) {
  return items.map(([key,label])=>`<option value="${esc(key)}"${String(key)===String(selected)?' selected':''}>${esc(label)}</option>`).join('');
}
function select(id, label, values, selected) {
  return `<label class="ol-select" for="${id}"><span>${label}</span><select id="${id}">${options(values,selected)}</select></label>`;
}
const downloadButton = text => `<button type="button" class="ol-button ol-download" data-ol-download disabled>${icon('download',16)} ${text}</button>`;
const action = (label, address) => `<a class="ol-button" href="#${address}">${label} ${icon('arrow',17)}</a>`;
const loading = message => `<div class="ol-state" role="status">${icon('refresh',23)}<p>${message}</p></div>`;
const empty = (title,message,link='') => `<div class="ol-state">${icon('info',24)}<h2>${esc(title)}</h2><p>${esc(message)}</p>${link}</div>`;

function header(tag, title, copy, side) {
  return `<header class="ol-intro">
    <div><div class="ol-kicker">${tag}</div><h1>${title}</h1><p>${copy}</p></div>
    ${side ? `<div class="ol-intro-action">${side}</div>` : ''}
  </header>`;
}

function isObserved(row, indicator) {
  const observed = row[indicator+'_is_observed'] ?? row.is_observed;
  return observed === true && isNumber(row[indicator]) && (row.valid_pixel_fraction == null || +row.valid_pixel_fraction >= .5)
    && (indicator !== 'ndwi' || (+row[indicator] >= -1 && +row[indicator] <= 1));
}

function observedSeries(data, zone, indicator) {
  const byDate = new Map();
  for (const row of data.observations || []) {
    if (zone !== 'all' && String(row.zone_id)!==zone || !isObserved(row,indicator)) continue;
    const date = datePart(row.date);
    const item = byDate.get(date) || {date,sum:0,n:0};
    item.sum += +row[indicator];
    item.n++;
    byDate.set(date,item);
  }
  return [...byDate.values()].sort((a,b)=>a.date.localeCompare(b.date)).map(item=>({date:item.date,value:item.sum/item.n}));
}

function latestWeights(data, indicator, latest) {
  const cells = new Map();
  for (const row of data.observations || []) {
    if (datePart(row.date)!==latest || !isObserved(row,indicator)) continue;
    const id=String(row.zone_id);
    if (!cells.has(id)) cells.set(id,new Set());
    cells.get(id).add(row.cell_id ?? `${row.lat},${row.lon}`);
  }
  return new Map([...cells].map(([id,keys])=>[id,keys.size]));
}

function projectionSeries(result, data, zone, indicator, latest) {
  let all = result.filter(item=>item.parameter===indicator && isNumber(item.prediction));
  if(zone!=='all') all=all.filter(item=>String(item.zone_id)===zone);
  else if(all.some(item=>item.zone_id==='all')) all=all.filter(item=>item.zone_id==='all');
  else {
    const weights=latestWeights(data,indicator,latest);
    const byDate=new Map();
    for (const row of all) {
      const weight=weights.get(String(row.zone_id)) ?? 0;
      if (!weight) continue;
      const date=datePart(row.date), current=byDate.get(date) || {date,weight:0,sum:0,experimental:false};
      current.weight+=weight;
      current.sum+=weight * +row.prediction;
      current.experimental ||= !!(row.is_extrapolated || row.is_extrapolation);
      byDate.set(date,current);
    }
    return [...byDate.values()].sort((a,b)=>a.date.localeCompare(b.date)).map(item=>({...item,value:item.sum/item.weight}));
  }
  return all.map(row=>({date:datePart(row.date),value:+row.prediction,
    lower:zone!=='all'&&isNumber(row.lower)?+row.lower:null,
    upper:zone!=='all'&&isNumber(row.upper)?+row.upper:null,
    experimental:!!(row.is_extrapolation || row.is_extrapolated)
  })).sort((a,b)=>a.date.localeCompare(b.date));
}

function chart(lines, opts={}) {
  const points = lines.flatMap(line=>line.points).filter(point=>isNumber(point.value) && Number.isFinite(timestamp(point.date)));
  if(!points.length) return empty('No chart for this selection','Choose a scope or indicator with available readings.');
  const width=810,height=274,pad={left:63,right:25,top:25,bottom:38};
  const axisBottom=height-pad.bottom, axisRight=width-pad.right;
  let start=Math.min(...points.map(p=>timestamp(p.date))), end=Math.max(...points.map(p=>timestamp(p.date)));
  if(start===end){start-=dayMs;end+=dayMs;}
  const values=points.map(p=>p.value), domains=(opts.band || []).filter(p=>isNumber(p.lower)&&isNumber(p.upper));
  for (const point of domains) values.push(point.lower,point.upper);
  const min=Math.min(...values),max=Math.max(...values);
  const gap=(max-min || Math.max(Math.abs(min)*.1,.1))*.15;
  const low=min-gap,high=max+gap;
  const x=value=>pad.left + (timestamp(value)-start)/(end-start)*(axisRight-pad.left);
  const y=value=>pad.top + (high-value)/(high-low)*(axisBottom-pad.top);
  const guides=Array.from({length:5},(_,i)=>{
    const ratio=i/4, line=pad.top+(axisBottom-pad.top)*ratio;
    return `<line class="ol-chart-grid" x1="${pad.left}" y1="${line}" x2="${axisRight}" y2="${line}"/><text class="ol-chart-tick" text-anchor="end" x="${pad.left-12}" y="${line+4}">${number(high-(high-low)*ratio,opts.precision??2)}</text>`;
  }).join('');
  const ticks=Array.from({length:5},(_,i)=>{
    const at=i/4, date=new Date(start+(end-start)*at).toISOString().slice(0,10), pos=pad.left + at*(axisRight-pad.left);
    return `<text class="ol-chart-tick" x="${pos}" y="${height-13}" text-anchor="${i===0?'start':i===4?'end':'middle'}">${shortDate(date)}</text>`;
  }).join('');
  const displayArea=opts.cutover && timestamp(opts.cutover)>=start && timestamp(opts.cutover)<=end
    ? `<rect class="ol-projected-area" x="${x(opts.cutover)}" y="${pad.top}" width="${Math.max(0,axisRight-x(opts.cutover))}" height="${axisBottom-pad.top}"/><line class="ol-cutover" x1="${x(opts.cutover)}" x2="${x(opts.cutover)}" y1="${pad.top}" y2="${axisBottom}"/>` : '';
  const band=domains.length>1?`<polygon class="ol-band" points="${[...domains.map(p=>`${x(p.date)},${y(p.lower)}`),...domains.slice().reverse().map(p=>`${x(p.date)},${y(p.upper)}`)].join(' ')}"/>`:'';
  const paths=lines.map(line=>{
    const usable=line.points.filter(p=>isNumber(p.value)&&Number.isFinite(timestamp(p.date)));
    const d=usable.map((p,i)=>`${i===0||(line.role==='observed'||line.role==='actual')&&timestamp(p.date)-timestamp(usable[i-1].date)>21*dayMs?'M':'L'}${x(p.date).toFixed(2)},${y(p.value).toFixed(2)}`).join(' ');
    return `<path class="ol-stroke ol-${line.role}" d="${d}"/>${usable.length===1 || line.showPoints && usable.length<110?usable.map(p=>`<circle class="ol-point ol-${line.role}" cx="${x(p.date)}" cy="${y(p.value)}" r="2.4"><title>${esc(formatDate(p.date))} · ${number(p.value,4)}</title></circle>`).join(''):''}`;
  }).join('');
  const selected=lines.flatMap(line=>line.role==='estimate'?line.points:[]).find(p=>p.date===opts.selected && isNumber(p.value));
  const marker=selected?`<line class="ol-selection" x1="${x(selected.date)}" x2="${x(selected.date)}" y1="${pad.top}" y2="${axisBottom}"/><circle class="ol-focus-point" cx="${x(selected.date)}" cy="${y(selected.value)}" r="5.8"/>`:'';
  return `<svg class="ol-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(opts.label||'Measurements over time')}">${displayArea}${guides}${ticks}${band}${paths}${marker}</svg>`;
}

function metric(rows) {
  const pairs=rows.filter(row=>isNumber(row.actual)&&isNumber(row.prediction));
  if (!pairs.length) return {n:0,rmse:null,mae:null,r2:null};
  const n=pairs.length, mean=pairs.reduce((sum,row)=>sum + +row.actual,0)/n;
  const square=pairs.reduce((sum,row)=>sum+(+row.prediction - +row.actual)**2,0);
  const variance=pairs.reduce((sum,row)=>sum+(+row.actual-mean)**2,0);
  return {n,rmse:Math.sqrt(square/n),mae:pairs.reduce((sum,row)=>sum+Math.abs(+row.prediction - +row.actual),0)/n,r2:variance>0?1-square/variance:null};
}

function callout({iconName='info',title,text}){
  return `<div class="ol-context">${icon(iconName,18)}<div><strong>${esc(title)}</strong><p>${esc(text)}</p></div></div>`;
}

async function renderLakeSummaries(host, data, signal, historical) {
  if (!host || (data.dataset || 'modis') !== 'modis') return;
  const fingerprint = data.model_benchmark?.fingerprint || data.static_bundle?.fingerprint;
  if (!fingerprint) return;
  try {
    const response = await fetch(new URL('./lake-summary-results.json', import.meta.url), {signal});
    if (!response.ok) return;
    const result = await response.json();
    if (signal?.aborted || !host.isConnected || result.fingerprint !== fingerprint) return;
    const heading = historical ? 'Lake median accuracy' : 'Lake median forecasts';
    host.className = 'ol-plot-panel ol-lake-summary';
    host.innerHTML = `<header class="ol-panel-head"><div><h2>${heading}</h2><p>${historical
      ? 'Each score compares a separately defined median with its observed value. The original mean forecasts are evaluated above.'
      : 'Projections start after each series’ final saved observation. The satellite archive ends in 2023.'}</p></div></header>
      <div class="ol-lake-controls">${select('ol-lake-series','Lake summary',result.series.map((row,index)=>[index,row.label]),0)}
      ${historical ? select('ol-lake-days','Forecast lead',benchmarkHorizons.map(days=>[days,`${days} days`]),8) : ''}
      ${downloadButton(historical ? 'Export actual and forecast pairs' : 'Export projections')}</div>
      <div data-ol-lake-summary-body></div>`;
    const seriesSelect = el(host,'#ol-lake-series'), daysSelect = el(host,'#ol-lake-days');
    const body = el(host,'[data-ol-lake-summary-body]'), button = el(host,'[data-ol-download]');
    function show() {
      const series = result.series[+seriesSelect.value];
      const rows = historical ? series.pairs.filter(row=>row.horizon_days===+daysSelect.value) : series.forecasts;
      const notes = `<p>${esc(series.basis)} · ${esc(series.unit)}</p>`;
      if (historical) {
        const scores = series.metrics.map(score=>`<div><span>R² · ${score.horizon_days} days</span><strong>${number(score.r2,4)}</strong><small>${score.n} pairs</small></div>`).join('');
        body.innerHTML = `${notes}<div class="ol-metrics">${scores}</div><div class="ol-plot-surface">${chart([
          {role:'actual',points:rows.map(row=>({date:row.target_date,value:row.actual})),showPoints:true},
          {role:'estimate',points:rows.map(row=>({date:row.target_date,value:row.prediction})),showPoints:true},
        ],{label:`${series.label} observed values and historical forecasts`})}<div class="ol-legend"><span><i></i> Observed median</span><span><i class="ol-legend-forecast"></i> Forecast</span></div></div>`;
      } else {
        const lastObserved=series.latest_observation;
        const observed=isNumber(lastObserved?.value)
          ? `Latest observed median ${number(lastObserved.value,3)} ${esc(series.unit)} · latest input ${esc(formatDate(lastObserved.last_sample_date || lastObserved.date))}.`
          : 'Latest observed median unavailable.';
        body.innerHTML = `${notes}<p>${observed}</p><div class="ol-metrics">${rows.map(row=>{
          const change=!isNumber(row.change) ? '' : row.change===0 ? 'Matches latest median'
            : `${number(Math.abs(row.change),3)} ${series.unit} ${row.change>0?'higher':'lower'} than latest median`;
          const score=series.metrics.find(candidate=>+candidate.horizon_days===+row.horizon_days);
          return `<div><span>${row.horizon_days} days · ${formatDate(row.target_date)}</span><strong>${number(row.prediction,3)}</strong>
            ${row.status ? `<small><b>${esc(row.status)}</b> · ${esc(row.reason || '')}</small>` : ''}
            ${change ? `<small>${esc(change)}</small>` : ''}<small>${esc(historicalSkill(score,row.horizon_days))}</small></div>`;
        }).join('')}</div>`;
      }
      button.disabled = !rows.length;
      button.onclick = () => download(toCsv(rows.map(row=>({...row,indicator:series.label,unit:series.unit}))),
        historical ? 'lake-median-accuracy.csv' : 'lake-median-forecast.csv','text/csv;charset=utf-8');
    }
    seriesSelect.addEventListener('change',show);
    daysSelect?.addEventListener('change',show);
    show();
  } catch (error) {
    if (!signal?.aborted && host.isConnected) {
      host.textContent = 'Lake median forecast results could not be loaded.';
    }
  }
}

function evidenceMarkup(result, saved) {
  const record=result?.explanation || result;
  if (!record) return '<p>No saved guidance is available for this indicator and location.</p>';
  const notes=(record.hypotheses || []).map(item=>`<li>${esc(item.statement || item.name || item.cause || item.hypothesis || '')}</li>`).filter(item=>item!=='<li></li>').slice(0,4);
  const sources=(record.sources || []).slice(0,6).map(item=>{
    const link=safeUrl(item.url);
    return `<li>${link ? `<a href="${esc(link)}" target="_blank" rel="noopener noreferrer">${esc(item.title || item.source_id)} ${icon('external',14)}</a>` : esc(item.title || item.source_id)}<p>${esc(item.text || item.summary || item.applicability || '')}</p></li>`;
  });
  return `<p>${esc(saved ? 'These saved references describe the indicator and location; they are not tied to the selected projection date.' : record.summary || 'Source context for this assessment.')}</p>
    ${notes.length?`<ul class="ol-notes-list">${notes.join('')}</ul>`:''}
    ${sources.length?`<ul class="ol-source-list">${sources.join('')}</ul>`:'<p>No matching source references were found.</p>'}`;
}

function historicalSkill(score, horizon) {
  if (!isNumber(score?.r2)) return `No historical score at about ${horizon} days`;
  const detail = score.r2 <= 0 ? 'Low historical skill' : score.r2 <= .2 ? 'Limited historical skill' : 'Positive historical skill';
  const pairs = isNumber(score.n) ? ` · ${number(score.n,0)} matched readings` : '';
  return `${detail} · R² ${number(score.r2,3)} at about ${horizon} days${pairs}`;
}

function conditionTone(status) {
  const value = String(status ?? '').toLowerCase();
  if (value.includes('unusual')) return 'unusual';
  if (value.includes('watch')) return 'watch';
  if (value.includes('typical') || value.includes('stable')) return 'typical';
  return 'neutral';
}

function conditionPanel({scope, origin, target, model, status, headline, summary, actionText, observedStatus, rows, coverageText}) {
  return `<header class="ol-condition-head"><div><p class="ol-kicker">SATELLITE OUTLOOK · ${esc(scope)}</p><h2>Projected condition</h2></div>
    <span class="ol-condition-badge" data-tone="${conditionTone(status)}">${esc(status)}</span></header>
    <dl class="ol-condition-times"><div><dt>Last observation</dt><dd>${esc(formatDate(origin))}${observedStatus ? ` · ${esc(observedStatus)}` : ''}</dd></div>
    <div><dt>Forecast for</dt><dd>${esc(formatDate(target))} · ${esc(model)}</dd></div></dl>
    <p class="ol-condition-summary">${headline ? `<strong>${esc(headline)}</strong> ` : ''}${esc(summary)}</p>
    <div class="ol-condition-signals">${rows.map(row=>`<article class="ol-condition-signal">
      <div><h3>${esc(row.label)}</h3><span data-tone="${conditionTone(row.status)}">${esc(row.status)}</span></div>
      <p class="ol-condition-readings">${isNumber(row.observed) ? `${number(row.observed,3)} observed${row.observedDate && row.observedDate!==origin ? ` on ${esc(formatDate(row.observedDate))}` : ''}` : 'Observed value unavailable'}
      <span aria-hidden="true">→</span> ${isNumber(row.projected) ? `${number(row.projected,3)} forecast` : 'Forecast unavailable'}${row.unit ? ` · ${esc(row.unit)}` : ''}</p>
      ${row.range ? `<small>${esc(row.range)}</small>` : ''}
      ${row.reason ? `<p>${esc(row.reason)}</p>` : ''}<small>${esc(row.skill)}</small>
    </article>`).join('')}</div>
    <footer class="ol-condition-footer">${actionText ? `<p><strong>Monitoring action</strong> ${esc(actionText)}</p>` : ''}
      ${coverageText ? `<small>${esc(coverageText)}</small>` : ''}
      <button type="button" class="ol-button" data-ol-condition-evidence>${icon('book',16)} Read supporting sources</button></footer>`;
}

export async function renderForecast(host,ctx) {
  const {data,mode,signal}=ctx, scopes=zones(data), modelOptions=models(data,mode,false);
  const current=ctx.state.forecast ||= {};
  const latest=datePart(data.report?.end || data.observations?.at(-1)?.date);
  const dataset=data.dataset || 'modis';
  const hasReference=mode==='api' && dataset==='modis' && data.model_benchmark?.saved_history && !!ctx.savedOutlook;
  if (hasReference && !modelOptions.some(([id])=>id==='auto')) {
    modelOptions.unshift(['auto',`Auto · ${data.model_benchmark.selected_model_name || 'recommended'}`]);
  }
  if (!Number.isFinite(timestamp(latest)) || !modelOptions.length) {
    host.innerHTML=`<div class="ol-view ol-forecast-view">
      ${header('SATELLITE MONITORING','Lake condition forecast','See projected changes in the satellite indicators and check how each forecast performed in the past.',action('Check accuracy','validation'))}
      ${empty('Lake forecast needs more observations','Open Data to review the measurement dates and available models for this upload.',action('Review dataset','data'))}
    </div>`;
    return;
  }
  if(!scopes.some(([key])=>key===current.zone)) current.zone='all';
  if(!modelOptions.some(([key])=>key===current.model)) current.model=modelOptions[0][0];
  if(!indicators.includes(current.indicator)) current.indicator='turbidity_proxy';
  if(!forecastHorizons.includes(current.days)) current.days=32;
  if(!Number.isFinite(current.offset) || current.offset<1 || current.offset>current.days) current.offset=current.days;
  if (![0,6,12,22].includes(current.history)) current.history=12;
  const indicatorsForSelect=indicators.map(id=>[id,parameterNames[id]]);
  host.innerHTML=`<div class="ol-view ol-forecast-view">
    ${header('SATELLITE MONITORING','Lake condition forecast','See what may change after the last satellite observation. Choose an 8, 16, or 32 day outlook to review monitoring signals and their historical skill.',action('Check accuracy','validation'))}
    <div class="ol-shell">
      <section class="ol-controls" aria-label="Forecast controls">
        ${select('ol-f-zone','Location',scopes,current.zone)}
        ${select('ol-f-model','Forecast model',modelOptions,current.model)}
        ${select('ol-f-interval','Outlook length',forecastHorizons.map(days=>[days,`${days} days${days>32?' · experimental':''}`]),current.days)}
        ${select('ol-f-indicator','Chart indicator',indicatorsForSelect,current.indicator)}
      </section>
      <div data-ol-forecast-content></div>
    </div>
    <section data-ol-lake-summary></section>
  </div>`;
  void renderLakeSummaries(el(host,'[data-ol-lake-summary]'),data,signal,false);
  const content=el(host,'[data-ol-forecast-content]');
  el(host,'.ol-intro-action a')?.addEventListener('click',()=>{
    const accuracy=ctx.state.accuracy ||= {};
    accuracy.zone=current.zone;
    accuracy.indicator=current.indicator;
    accuracy.days=benchmarkHorizons.reduce((best,days)=>Math.abs(days-current.offset)<Math.abs(best-current.offset)?days:best);
    if(models(data,mode,true).some(([key])=>key===current.model)) accuracy.model=current.model;
  });
  let requestNumber=0, forecast=null, lastWeightKey='', evidenceTimer, evidenceController, evidenceKey='';
  let monitoringRequest=null;
  let frame=0;
  const observedCache=new Map(), projections=new Map();
  const evidenceCache=new Map(), forecastCache=new Map(), monitoringCache=new Map();
  function cancelEvidence() {
    clearTimeout(evidenceTimer);
    evidenceController?.abort();
    evidenceController=null;
    evidenceKey='';
  }
  signal?.addEventListener('abort',()=>{
    cancelEvidence();
    monitoringRequest?.controller.abort();
    cancelAnimationFrame(frame);
  },{once:true});

  const getObserved=(zone,indicator)=>{
    const key=`${zone}:${indicator}`;
    if(!observedCache.has(key)) observedCache.set(key,observedSeries(data,zone,indicator));
    return observedCache.get(key);
  };
  const evidenceMarkupFor=record=>evidenceMarkup(record,mode==='snapshot');

  function getMonitoring(zone, model) {
    const key=JSON.stringify([zone,model]);
    if (monitoringCache.has(key)) return Promise.resolve(monitoringCache.get(key));
    if (monitoringRequest?.key===key) return monitoringRequest.promise;
    monitoringRequest?.controller.abort();
    const controller=new AbortController();
    const promise=ctx.request(url('monitoring',{dataset,zone,model,evidence_id:data.evidence_id}),{signal:controller.signal}).then(result=>{
      if (controller.signal.aborted) return null;
      monitoringCache.set(key,result);
      if (monitoringCache.size>12) monitoringCache.delete(monitoringCache.keys().next().value);
      return result;
    }).catch(()=>null).finally(()=>{
      if (monitoringRequest?.controller===controller) monitoringRequest=null;
    });
    monitoringRequest={key,controller,promise};
    return promise;
  }

  function draw() {
    if(!forecast || signal?.aborted) return;
    cancelEvidence();
    cancelAnimationFrame(frame);
    const scope=scopes.find(([key])=>key===current.zone)?.[1] || current.zone;
    const {metadata={},rows}=forecast;
    const model=metadata.selected_model_name || modelOptions.find(([key])=>key===current.model)?.[1] || niceModel(current.model);
    const effective=metadata.selected_model || (current.model==='auto'?data.model_benchmark?.selected_model:current.model);
    const directDaily=effective==='prophet';
    const prefix=`${current.model}:${current.days}:${current.zone}`;
    if(prefix!==lastWeightKey) {
      projections.clear();
      for(const id of indicators) projections.set(id,projectionSeries(rows,data,current.zone,id,latest));
      lastWeightKey=prefix;
    }
    const recorded=getObserved(current.zone,current.indicator);
    const baseline=recorded.at(-1);
    const projected=projections.get(current.indicator);
    const withBand=current.zone!=='all'&&projected.some(point=>isNumber(point.lower)&&isNumber(point.upper));
    const labels=indicator => parameterNames[indicator];
    const errors=(data.model_benchmark?.metrics || []).filter(row=>row.model===effective && row.parameter===current.indicator && String(row.zone_id)===current.zone);
    const below=errors.filter(row=>isNumber(row.r2) && +row.r2<0).sort((a,b)=>a.horizon_days-b.horizon_days);
    const preparedValue=number(baseline?.value,4);
    const signed=value=>`${value>0?'+':''}${number(value,4)}`;
    const end=addDays(latest,current.days);
    const savedProjection=mode==='snapshot'
      ? data.monitoring_projections?.[current.model]?.[current.zone] || data.monitoring_projections?.[effective]?.[current.zone]
      : null;
    let monitoring=mode==='snapshot' ? {
      observation:data.satellite_monitoring?.[current.zone],
      projection:savedProjection?.model===effective ? savedProjection : null,
    } : null;
    let monitoringPending=mode==='api';

    function renderProjectedCondition(target) {
      const card=el(content,'[data-ol-condition]');
      if (!card) return;
      const observed=monitoring?.observation;
      const projectedCondition=monitoring?.projection?.horizons?.find(item=>
        +item.horizon_days===current.offset && datePart(item.target_date)===target);
      const nativeDay=benchmarkHorizons.includes(current.offset);
      const experimental=current.offset>(metadata.max_validated_horizon_days || 32);
      const nearestHorizon=benchmarkHorizons.reduce((nearest,value)=>
        Math.abs(current.offset-value)<Math.abs(current.offset-nearest) ? value : nearest);
      const assessedRows=new Map((projectedCondition?.indicators || []).map(row=>[row.parameter,row]));
      const observedRows=new Map((observed?.indicators || []).map(row=>[row.parameter,row]));
      const parameters=[...new Set([...indicators,...assessedRows.keys()])];
      const conditionRows=parameters.map(parameter=>{
        const assessed=assessedRows.get(parameter);
        const observation=observedRows.get(parameter);
        const last=getObserved(current.zone,parameter).at(-1);
        const matchedObservation=isNumber(observation?.value);
        const point=projections.get(parameter)?.find(item=>item.date===target);
        const metric=(mode==='snapshot' || data.model_benchmark?.saved_history ? data.model_benchmark?.metrics || [] : [])
          .find(row=>row.model===effective && row.parameter===parameter && String(row.zone_id)===current.zone
            && +row.horizon_days===nearestHorizon);
        const score=isNumber(assessed?.r2) ? {r2:assessed.r2,n:assessed.pairs} : metric;
        const range=isNumber(assessed?.range_low) && isNumber(assessed?.range_high)
          ? `Recent reference range · ${number(assessed.range_low,3)} to ${number(assessed.range_high,3)}` : '';
        return {
          label:assessed?.label || observation?.label || labels(parameter) || parameter,
          observed:matchedObservation ? observation.value : last?.value,
          projected:assessed ? assessed.prediction : point?.value,
          observedDate:datePart(matchedObservation ? observation.date : last?.date),
          unit:assessed?.unit || observation?.unit,range,
          status:assessed?.status || (experimental ? 'Scenario' : !nativeDay ? (directDaily ? 'Daily estimate' : 'Interpolated') : monitoringPending ? 'Checking' : 'Unclassified'),
          reason:assessed?.reason,skill:experimental ? 'Beyond the evaluated forecast period' : historicalSkill(score,nearestHorizon),
        };
      });
      const status=projectedCondition?.status || (experimental ? 'Experimental scenario' : nativeDay
        ? (monitoringPending ? 'Assessing forecast' : 'Assessment unavailable') : (directDaily ? 'Daily forecast' : 'Interpolated forecast'));
      const summary=projectedCondition?.summary || (experimental
        ? 'This lead falls beyond the evaluated window. Choose a shorter outlook for monitoring signals.'
        : !nativeDay ? (directDaily ? 'Prophet estimates this calendar date directly.' : 'This day uses values between model forecast horizons.') + ' Choose day 8, 16, or 32 to check the projected condition.'
          : monitoringPending ? 'Checking the projected readings against the recent observed range and historical forecast skill.'
            : 'Compare the forecast values below with the latest recorded observations and the historical scores.');
      const today=new Date();
      today.setHours(0,0,0,0);
      const archive=new Date(`${target}T00:00:00`)<today ? 'Archive outlook. The projected date has already passed.'
        : observed?.source?.stale ? 'This forecast starts from an archived observation.' : '';
      const pixelQuality=parameters.some(parameter=>observedRows.get(parameter)?.coverage_unknown===true)
        ? 'Some readings lack pixel coverage information.' : '';
      const coverageText=[archive,pixelQuality].filter(Boolean).join(' ');
      card.innerHTML=conditionPanel({scope,origin:datePart(observed?.date || latest),target,model,status,
        headline:projectedCondition?.headline,summary,actionText:projectedCondition?.next_step,
        observedStatus:observed?.status,rows:conditionRows,coverageText});
      el(card,'[data-ol-condition-evidence]').onclick=()=>{
        const evidence=el(content,'[data-ol-evidence]');
        evidence.open=true;
        el(evidence,'summary')?.focus();
        evidence.scrollIntoView({block:'nearest'});
      };
    }

    const renderState=({updateEvidence=false}={}) => {
      const target=addDays(latest,current.offset);
      const estimate=projected.find(item=>item.date===target);
      const history=current.history ? recorded.slice(-current.history) : [];
      el(host,'[data-ol-selected-date]').textContent=formatDate(target);
      el(host,'[data-ol-selected-value]').textContent=number(estimate?.value,4);
      el(host,'[data-ol-forecast-date]').value=target;
      const both=isNumber(baseline?.value)&&isNumber(estimate?.value);
      el(host,'[data-ol-selected-difference]').textContent=both?signed(estimate.value-baseline.value):'—';
      const selectedState=el(host,'[data-ol-selected-state]');
      selectedState.textContent=current.offset>32?'Experimental':'Short outlook';
      selectedState.dataset.experimental=String(current.offset>32);
      el(host,'[data-ol-detail-date]').textContent=formatDate(target);
      const slider=el(host,'[data-ol-slider]');
      slider.value=current.offset;
      slider.setAttribute('aria-valuetext',`+${current.offset} days, ${formatDate(target)}`);
      el(host,'[data-ol-offset]').textContent=`+${current.offset} days`;
      for(const button of content.querySelectorAll('[data-ol-jump]')) {
        button.setAttribute('aria-pressed',String(+button.dataset.olJump===current.offset));
      }
      const anchor=baseline?.date===latest ? [baseline] : [];
      el(host,'[data-ol-plot]').innerHTML=chart([
        {points:history,role:'observed',showPoints:true},
        {points:current.history ? [...anchor,...projected] : projected,role:'estimate'},
      ],{label:`${labels(current.indicator)} history and forecast for ${scope}`,cutover:latest,band:withBand?projected:[],selected:target});
      el(host,'[data-ol-projection-cards]').innerHTML=indicators.map((id,index)=>{
        const point=projections.get(id)?.find(item=>item.date===target);
        const previous=getObserved(current.zone,id).at(-1);
        const interval=point && isNumber(point.lower) && isNumber(point.upper)? `Empirical range: ${number(point.lower,3)} to ${number(point.upper,3)}`: point ? 'No calibrated interval' : 'No estimate for this selection';
        const change=point&&isNumber(previous?.value) ? `${signed(point.value - previous.value)} vs last reading` : '';
        return `<article class="ol-estimate"><span class="ol-estimate-label">${icon(['waves','drop','spark'][index],16)} ${esc(shortNames[id])}</span><strong>${number(point?.value,4)}</strong><small>${esc(change || interval)}</small>${change?`<small>${esc(interval)}</small>`:''}</article>`;
      }).join('');
      renderProjectedCondition(target);
      const details=el(host,'[data-ol-evidence]');
      if (details.open && updateEvidence) scheduleEvidence(details,target);
    };

    function scheduleEvidence(details,target) {
      const inner=el(details,'[data-ol-evidence-content]');
      const key=JSON.stringify([target,current.zone,current.indicator,current.model]);
      if (evidenceCache.has(key)) {
        cancelEvidence();
        inner.innerHTML=evidenceMarkupFor(evidenceCache.get(key));
        return;
      }
      if(evidenceKey===key) return;
      cancelEvidence();
      evidenceKey=key;
      inner.innerHTML='<p>Retrieving source context…</p>';
      evidenceTimer=setTimeout(async()=>{
        const controller=new AbortController();
        evidenceController=controller;
        try {
          const record=mode==='snapshot'?data.explanations?.[`${current.zone}:${current.indicator}`]:await ctx.request(url('explain',{dataset,zone:current.zone,parameter:current.indicator,forecast_date:target,model:current.model,evidence_id:data.evidence_id}),{signal:controller.signal});
          if(!signal?.aborted&&!controller.signal.aborted&&evidenceKey===key&&details.isConnected) {
            evidenceCache.set(key,record);
            inner.innerHTML=evidenceMarkupFor(record);
          }
        } catch(error) {
          if(!signal?.aborted&&!controller.signal.aborted&&evidenceKey===key&&details.isConnected) {
            inner.innerHTML=`<p>${esc(error.message || 'No source context could be loaded.')}</p>`;
          }
        } finally {
          if(evidenceController===controller) {
            evidenceController=null;
            evidenceKey='';
          }
        }
      },220);
    }

    content.innerHTML=`<div class="ol-forecast-context" aria-label="Forecast basis">
      <div><span>Forecast starts after</span><strong>${esc(formatDate(latest))}</strong></div>
      <div><span>Outlook ends</span><strong>${esc(formatDate(end))}</strong></div>
      <div><span>Data source</span><strong>${esc(metadata.display_source || (mode==='snapshot'?'Saved MODIS forecast':'Local forecast engine'))}</strong></div>
    </div>
    <section class="ol-date-tools" aria-label="Forecast date controls">
      <label class="ol-select" for="ol-f-date"><span>Choose forecast date</span><input type="date" id="ol-f-date" data-ol-forecast-date min="${addDays(latest,1)}" max="${end}"></label>
      <div class="ol-date-jumps"><span>Jump to a day after the last observation</span><div>${benchmarkHorizons.map(days=>`<button type="button" class="ol-button" data-ol-jump="${days}" aria-pressed="false">+${days} days</button>`).join('')}</div></div>
    </section>
    <section class="ol-condition" aria-label="Projected lake condition" data-ol-condition></section>
    <div class="ol-forecast-layout">
      <section class="ol-plot-panel">
        <header class="ol-panel-head">
          <div><div class="ol-kicker ol-kicker-light">OBSERVED AND PREDICTED</div><h2>${esc(labels(current.indicator))}</h2><p>${esc(String(scope))} · ${esc(model)}</p></div>
          ${downloadButton('Export CSV')}
        </header>
        <div class="ol-readouts"><div><span>Last recorded · ${formatDate(baseline?.date)}</span><strong>${preparedValue}</strong></div>
        <div><span>Estimate · <span data-ol-selected-date></span></span><strong data-ol-selected-value>—</strong></div>
        <div><span>Difference from last reading</span><strong data-ol-selected-difference>—</strong></div></div>
        <div class="ol-chart-tools">${select('ol-f-history','Chart window',[[12,'Last 12 readings and forecast'],[6,'Last 6 readings and forecast'],[22,'Last 22 readings and forecast'],[0,'Forecast only']],current.history)}</div>
        <div class="ol-plot-surface"><div data-ol-plot></div>
          <div class="ol-legend"><span data-ol-observed-key><i></i> Actual readings</span><span><i class="ol-legend-forecast"></i> Forecast</span>${withBand?'<span><i class="ol-legend-band"></i> Empirical range</span>':''}</div>
        </div>
        <div class="ol-time-controls">
          <label for="ol-scrub"><span>Inspect a day</span><output data-ol-offset></output></label>
          <input id="ol-scrub" data-ol-slider type="range" min="1" max="${current.days}" step="1">
          <div class="ol-time-dates"><span>Day 1</span><span>Day ${current.days}</span></div>
        </div>
      </section>
      <aside class="ol-detail-panel">
        <div class="ol-kicker">DAY DETAIL</div>
        <h2>All three estimates</h2>
        <div class="ol-detail-date"><time data-ol-detail-date></time><span class="ol-forecast-tag" data-ol-selected-state></span></div>
        <div class="ol-estimates" data-ol-projection-cards></div>
        <p class="ol-unit-note">Water index is dimensionless. Turbidity and chlorophyll are satellite proxies, not measured concentrations.</p>
      </aside>
    </div>
    ${current.days>32 ? callout({iconName:'alert',title:'Experimental beyond 32 days',text:'Longer projections extend the short outlook. They have no validated accuracy or uncertainty bands.'}) : ''}
    ${below.length ? callout({iconName:'alert',title:'Limited historical skill',text:`This model has a negative R² for this indicator and location at ${below.map(item=>`about ${item.horizon_days} days`).join(', ')}.`}) : ''}
    <div class="ol-methods">
      <details><summary>${icon('info',17)} How to read the outlook <span>${icon('down',17)}</span></summary><div>
        <p>${directDaily?'Prophet estimates each calendar date directly. Historical comparisons cover roughly 8, 16 and 32 days.':'Daily estimates interpolate model outputs near 8, 16 and 32 days.'} ${current.zone==='all'?'Whole-lake estimates have no calibrated uncertainty interval.':'Zone bands illustrate historical residuals; their future coverage is unverified.'}</p>
        <p>${esc(metadata.uncertainty || 'Estimates are exploratory. Review the retrospective results before interpreting the forecast.')}</p>
      </div></details>
      <details data-ol-evidence><summary>${icon('book',17)} Sources and context <span>${icon('down',17)}</span></summary><div data-ol-evidence-content></div></details>
    </div>`;
    const slider=el(host,'[data-ol-slider]');
    const evidence=el(host,'[data-ol-evidence]');
    slider.addEventListener('input',()=>{
      current.offset=+slider.value;
      if (evidence.open) {
        cancelEvidence();
        el(evidence,'[data-ol-evidence-content]').textContent='Source context updates after you choose a day.';
      }
      cancelAnimationFrame(frame);
      frame=requestAnimationFrame(()=>renderState());
    });
    slider.addEventListener('change',()=>{
      cancelAnimationFrame(frame);
      renderState({updateEvidence:true});
    });
    const dateInput=el(host,'[data-ol-forecast-date]');
    dateInput.addEventListener('change',()=>{
      const next=Math.round((timestamp(dateInput.value)-timestamp(latest))/dayMs);
      if(next>=1 && next<=current.days) current.offset=next;
      renderState({updateEvidence:true});
    });
    for(const button of content.querySelectorAll('[data-ol-jump]')) {
      button.onclick=()=>{
        const days=+button.dataset.olJump;
        current.offset=days;
        if(current.days<days) {
          current.days=days;
          el(host,'#ol-f-interval').value=days;
          return load();
        }
        renderState({updateEvidence:true});
      };
    }
    const chartWindow=el(host,'#ol-f-history');
    const updateChartWindow=()=>{
      el(host,'[data-ol-observed-key]').hidden=!current.history;
    };
    chartWindow.addEventListener('change',()=>{
      current.history=+chartWindow.value;
      updateChartWindow();
      renderState();
    });
    evidence.addEventListener('toggle',()=>{
      if(evidence.open) scheduleEvidence(evidence,addDays(latest,current.offset));
      else cancelEvidence();
    });
    const exportButton=el(host,'[data-ol-download]');
    exportButton.disabled=false;
    exportButton.addEventListener('click',()=>{
      const selection=indicators.flatMap(id=>(projections.get(id)||[]).map(point=>({date:point.date,zone_id:current.zone,model:effective || current.model,parameter:id,prediction:point.value,lower:point.lower??null,upper:point.upper??null,experimental:!!point.experimental})));
      if(!selection.length){ctx.notify?.('No estimates to export.','warning');return;}
      download(toCsv(selection),'aqualens-outlook.csv','text/csv;charset=utf-8');
    });
    updateChartWindow();
    renderState();
    if (mode==='api') {
      const conditionNode=el(content,'[data-ol-condition]');
      void getMonitoring(current.zone,current.model).then(result=>{
        if (signal?.aborted || !conditionNode?.isConnected) return;
        monitoring=result;
        monitoringPending=false;
        renderProjectedCondition(addDays(latest,current.offset));
      });
    }
  }

  async function load() {
    const turn=++requestNumber, model=current.model, days=current.days;
    cancelEvidence();
    if (monitoringRequest) monitoringRequest.controller.abort();
    monitoringRequest=null;
    cancelAnimationFrame(frame);
    const technique=modelOptions.find(([key])=>key===model)?.[1] || niceModel(model);
    const cacheKey=JSON.stringify([model,days]);
    content.innerHTML=`<div class="ol-state ol-forecast-loading" role="status"><span class="ol-spinner" aria-hidden="true"></span>
      <h2>Loading the ${esc(days)}-day forecast</h2><p>${esc(technique)} · Starting from ${esc(formatDate(latest))}.</p>
      <p>You can still adjust the forecast controls.</p></div>`;
    forecast=null;
    try {
      let result=forecastCache.get(cacheKey);
      if(!result && mode==='snapshot') {
        const rows=data.forecasts?.[model];
        result={rows:Array.isArray(rows)?rows:rows?.rows,metadata:data.forecast_metadata?.[model] || rows?.metadata};
      } else if(!result) {
        result=await ctx.savedOutlook?.('forecast',{model,days});
        if (!result && !signal?.aborted && turn===requestNumber) result=await ctx.request(url('forecast',{dataset,model,horizon_days:days}),{signal});
      }
      if(signal?.aborted||turn!==requestNumber) return;
      if(!result?.rows?.length) {
        showUnavailable('No forecast is ready','No estimates were returned for this model and dataset.');
        return;
      }
      forecastCache.set(cacheKey,result);
      if (forecastCache.size>4) forecastCache.delete(forecastCache.keys().next().value);
      const last=addDays(latest,days);
      forecast={...result,rows:result.rows.filter(row=>datePart(row.date)<=last && datePart(row.date)>latest)};
      lastWeightKey='';
      draw();
    } catch(error) {
      if(!signal?.aborted&&turn===requestNumber) showUnavailable('Forecast unavailable',error.message || 'Unable to load these estimates.');
    }
  }
  function showUnavailable(title,message) {
    content.innerHTML=empty(title,message,`<div class="ol-forecast-recovery"><button type="button" class="ol-button" data-ol-retry>Retry forecast</button>${action('Open data and training','data')}</div>`);
    el(content,'[data-ol-retry]').onclick=load;
  }
  const fields={'ol-f-zone':['zone',false],'ol-f-model':['model',true],'ol-f-interval':['days',true],'ol-f-indicator':['indicator',false]};
  for(const [id,[field,reload]] of Object.entries(fields)) {
    el(host,`#${id}`).addEventListener('change',event=>{
      current[field]=field==='days'?+event.target.value:event.target.value;
      if(field==='days') current.offset=Math.min(current.offset,current.days);
      if(reload) load();
      else draw();
    });
  }
  await load();
}

export async function renderAccuracy(host,ctx) {
  const {data,mode,signal}=ctx, scopes=zones(data);
  const modelOptions=models(data,mode,true), current=ctx.state.accuracy ||= {};
  if(!scopes.some(([key])=>key===current.zone)) current.zone='all';
  if(!modelOptions.some(([key])=>key===current.model)) current.model=modelOptions[0]?.[0] || 'det';
  if(!benchmarkHorizons.includes(current.days)) current.days=8;
  if(!indicators.includes(current.indicator)) current.indicator='ndwi';
  const dataset=data.dataset || 'modis';
  host.innerHTML=`<div class="ol-view ol-accuracy-view">
    ${header('HISTORICAL EVALUATION','Forecast accuracy','Compare past predictions with their recorded observations. Open any indicator and lead time to inspect the exact pairs.',action('Open forecast','forecasts'))}
    <section class="ol-shell">
      <div class="ol-controls" aria-label="Accuracy filters">
        ${select('ol-a-zone','Location',scopes,current.zone)}
        ${select('ol-a-model','Historical model',modelOptions.length?modelOptions:[['det','DET']],current.model)}
        ${select('ol-a-horizon','Forecast lead',benchmarkHorizons.map(days=>[days,`About ${days} days`]),current.days)}
        ${select('ol-a-indicator','Chart indicator',indicators.map(id=>[id,parameterNames[id]]),current.indicator)}
      </div>
      <div data-ol-accuracy-content></div>
    </section>
    <section data-ol-lake-summary></section>
  </div>`;
  void renderLakeSummaries(el(host,'[data-ol-lake-summary]'),data,signal,true);
  const content=el(host,'[data-ol-accuracy-content]');
  let turn=0, result=null;
  const cache=new Map();
  const periodLabel=(from,to)=>from&&to?`${formatDate(from)} – ${formatDate(to)}`:'Period not supplied';
  const validDate=value=>/^\d{4}-\d{2}-\d{2}$/.test(value||'')&&Number.isFinite(timestamp(value));
  const modelLabel=id=>data.model_benchmark?.models?.find(item=>item.id===id)?.name||niceModel(id);
  function stateCard(title,copy,{retry=false,related=''}={}) {
    content.innerHTML=`<div class="ol-state" role="status">${icon('info',24)}<h2>${esc(title)}</h2><p>${esc(copy)}</p>
      <div class="ol-a-actions">${retry?'<button class="ol-button" type="button" data-ol-a-retry>Try again</button>':''}${related}</div></div>`;
    el(content,'[data-ol-a-retry]')?.addEventListener('click',load);
  }

  function draw() {
    if(!result || signal?.aborted) return;
    const rows=result.rows.filter(row=>String(row.zone_id)===current.zone && isNumber(row.actual)&&isNumber(row.prediction));
    const cells=Object.fromEntries(indicators.map(id=>[id,Object.fromEntries(benchmarkHorizons.map(days=>[days,metric(rows.filter(row=>row.parameter===id&&+row.horizon_days===days))]))]));
    const goal=.20, targets=Object.values(cells).flatMap(value=>Object.values(value)), passed=targets.filter(value=>isNumber(value.r2)&&value.r2>goal).length;
    const scope=scopes.find(([key])=>key===current.zone)?.[1] || current.zone;
    const model=result.model_name || result.metadata?.selected_model_name || modelOptions.find(([key])=>key===current.model)?.[1] || niceModel(current.model);
    const modelId=result.model || result.metadata?.selected_model || (current.model==='auto'?data.model_benchmark?.selected_model:current.model);
    const benchmark=data.model_benchmark?.metrics || [];
    const comparable=benchmark.filter(row=>row.parameter===current.indicator&&+row.horizon_days===current.days&&String(row.zone_id)===current.zone&&isNumber(row.rmse)).sort((a,b)=>+a.rmse - +b.rmse);
    const periods=result.metadata || {};
    const availableDates=rows.map(row=>datePart(row.target_date)).filter(validDate).sort();
    const first=datePart(periods.test_start)||availableDates[0], last=datePart(periods.test_end)||availableDates.at(-1);
    if(!validDate(current.from)||current.from<first||current.from>last) current.from=first;
    if(!validDate(current.to)||current.to>last||current.to<first||current.to<current.from) current.to=last;
    const rangeActive=current.from!==first||current.to!==last;
    const comparison=periodLabel(first,last);
    const scopedRows=rows.filter(item=>item.parameter===current.indicator && +item.horizon_days===current.days &&
      datePart(item.target_date)>=current.from&&datePart(item.target_date)<=current.to)
      .sort((a,b)=>datePart(a.target_date).localeCompare(datePart(b.target_date)));
    const scores=metric(scopedRows);
    const selected=scopedRows.find(item=>datePart(item.target_date)===current.date)||scopedRows.at(-1);
    const selectedIndex=scopedRows.indexOf(selected);
    current.date=selected?datePart(selected.target_date):'';
    const error=selected?+selected.prediction - +selected.actual:null;
    const errorLabel=error===null?'No reading selected':error>0?'Estimate was above the observation':error<0?'Estimate was below the observation':'Estimate matched the observation';
    const matrix=indicators.map(id=>`<tr><th scope="row">${esc(shortNames[id])}</th>${benchmarkHorizons.map(days=>{
      const value=cells[id][days], active=current.indicator===id&&current.days===days;
      const status=value.r2==null?'No score':value.r2>goal?'Above goal':value.r2<0?'Negative':'Below goal';
      return `<td><button data-ol-cell data-indicator="${id}" data-days="${days}" type="button" class="ol-score ${value.r2!=null && value.r2>goal?'met':''}${active?' selected':''}" aria-pressed="${active}" aria-label="${esc(shortNames[id])}, about ${days} days, R squared ${number(value.r2,3)}, ${status}"><strong>${number(value.r2,3)}</strong><small>${status}</small></button></td>`;
    }).join('')}</tr>`).join('');
    const referenceLabel=periods.display_source || 'Frozen historical prediction pairs';
    content.innerHTML=`<section class="ol-a-period" aria-label="Evaluation period and source">
      <div><span class="ol-kicker">EVALUATION TARGET DATES</span><strong>${esc(comparison)}</strong><small>${esc(referenceLabel)}</small></div>
      <div><span class="ol-kicker">FIT THROUGH</span><strong>${formatDate(periods.train_end)}</strong><small>Earlier observations only</small></div>
      <div><span class="ol-kicker">MODEL SELECTION THROUGH</span><strong>${formatDate(periods.validation_end)}</strong><small>Before the evaluated observations</small></div>
    </section>
    <form class="ol-a-range" aria-label="Chart date range" data-ol-a-range>
      <div class="ol-a-range-caption"><strong>Chart date range</strong><small>Filters the chart, errors, and export.</small></div>
      <label class="ol-select" for="ol-a-from"><span>From</span><input id="ol-a-from" name="from" type="date" required min="${esc(first)}" max="${esc(last)}" value="${esc(current.from)}"></label>
      <label class="ol-select" for="ol-a-to"><span>Through</span><input id="ol-a-to" name="to" type="date" required min="${esc(first)}" max="${esc(last)}" value="${esc(current.to)}"></label>
      <div class="ol-a-actions">
        <button class="ol-button ol-a-primary" type="submit">Apply dates</button>
        <button class="ol-button" type="button" data-ol-a-reset ${rangeActive?'':'disabled'}>Full period</button>
      </div>
      <p class="ol-a-form-error" data-ol-a-form-error role="status"></p>
    </form>
    <div class="ol-accuracy-plot">
      <section class="ol-plot-panel ol-accuracy-chart" aria-labelledby="ol-a-plot-title">
        <header class="ol-panel-head"><div><div class="ol-kicker">OBSERVED VS FORECAST</div><h2 id="ol-a-plot-title" tabindex="-1">${esc(parameterNames[current.indicator])}</h2>
          <p>${esc(scope)} · ${esc(model)} · About ${current.days} days ahead</p></div>${downloadButton(`Export ${scores.n} pairs`)}</header>
        <div class="ol-metrics"><div><span>R²</span><strong>${number(scores.r2,3)}</strong></div><div><span>Typical error (MAE)</span><strong>${number(scores.mae,4)}</strong></div><div><span>RMSE</span><strong>${number(scores.rmse,4)}</strong></div><div><span>Matched readings</span><strong>${number(scores.n,0)}</strong></div></div>
        <p class="ol-a-window">${esc(periodLabel(current.from,current.to))}${rangeActive?' · Filtered dates':' · Full evaluation period'}</p>
        <div class="ol-plot-surface" data-ol-a-plot>${scopedRows.length?chart([
          {role:'actual',points:scopedRows.map(item=>({date:datePart(item.target_date),value:+item.actual})),showPoints:true},
          {role:'estimate',points:scopedRows.map(item=>({date:datePart(item.target_date),value:+item.prediction})),showPoints:true}
        ],{label:`Actual values and model forecasts for ${parameterNames[current.indicator]}`,selected:current.date}):empty('No matched readings in these dates','Widen the range or choose another indicator or lead time.',rangeActive?'<button class="ol-button" type="button" data-ol-a-reset>Show full period</button>':'')}
        ${scopedRows.length?'<div class="ol-legend"><span><i></i> Actual reading</span><span><i class="ol-legend-forecast"></i> Historical forecast</span></div>':''}</div>
        ${scopedRows.length?'<p class="ol-a-plot-help">Select the chart or use the date picker to inspect a matched reading.</p>':''}
        ${scores.n>0&&scores.r2===null?'<p class="ol-a-plot-help">R² requires more than one distinct observation value. Typical error and RMSE are still shown.</p>':''}
      </section>
      <aside class="ol-detail-panel ol-pair">
        <div class="ol-kicker">PAIRED READING</div><h2>Inspect a date</h2>
        ${select('ol-a-date','Target observation',scopedRows.map(row=>[datePart(row.target_date),formatDate(row.target_date)]),current.date)}
        <div class="ol-a-step" aria-label="Move between observations">
          <button class="ol-button" type="button" data-ol-a-step="-1" ${selectedIndex<=0?'disabled':''}>Previous</button>
          <span>${selected?selectedIndex+1:0} of ${scores.n}</span>
          <button class="ol-button" type="button" data-ol-a-step="1" ${selectedIndex<0||selectedIndex>=scores.n-1?'disabled':''}>Next</button>
        </div>
        <dl><div><dt>Actual observation</dt><dd>${number(selected?.actual,4)}</dd></div><div><dt>Model estimate</dt><dd>${number(selected?.prediction,4)}</dd></div>
          <div><dt>Error (estimate − actual)</dt><dd class="ol-a-error-value">${error>0?'+':''}${number(error,4)}</dd></div><div><dt>Forecast made on</dt><dd>${formatDate(selected?.origin_date)}</dd></div>
          <div><dt>Actual time ahead</dt><dd>${isNumber(selected?.actual_elapsed_days)?`${number(selected.actual_elapsed_days,0)} days`:'—'}</dd></div>
        </dl>
        <small>${esc(errorLabel)}. Satellite observation dates can be unevenly spaced.</small>
      </aside>
    </div>
    ${scores.r2!=null&&scores.r2<0 ? callout({iconName:'alert',title:'This forecast has a negative R²',text:'It had greater squared error than simply using the average of these recorded values.'}) : ''}
    <section class="ol-scoreboard" aria-label="Accuracy goal scorecard">
      <div class="ol-score-summary"><div class="ol-kicker ol-kicker-light">FULL PERIOD · R² GOAL</div><strong>${passed}<span> / 9</span></strong><h2>scores above 0.20</h2>
        <p>${esc(model)} · ${esc(scope)}</p><small>${esc(comparison)}</small></div>
      <div class="ol-score-detail"><div><div class="ol-kicker">R² BY INDICATOR AND LEAD</div><h2>Where was the forecast stronger?</h2><p>These scores always use the full evaluation period. Select a score to open its chart.</p></div>
        <table><thead><tr><th scope="col">Indicator</th>${benchmarkHorizons.map(days=>`<th scope="col">~${days} days</th>`).join('')}</tr></thead><tbody>${matrix}</tbody></table>
      </div>
    </section>
    ${comparable.length?`<section class="ol-a-benchmark" aria-labelledby="ol-a-model-comparison">
      <header><div class="ol-kicker">FULL PERIOD · MODEL COMPARISON</div><h2 id="ol-a-model-comparison">Compare candidate models</h2>
        <p>${esc(parameterNames[current.indicator])} · ${esc(scope)} · About ${current.days} days. Lower RMSE means smaller errors on these matched readings.</p></header>
      <div class="ol-compare" tabindex="0" aria-label="Candidate model comparison">
        <table><thead><tr><th scope="col">Model</th><th scope="col">RMSE</th><th scope="col">R²</th><th scope="col">Pairs</th></tr></thead><tbody>${comparable.map(row=>{
          const currentModel=row.model===modelId, canOpen=modelOptions.some(([key])=>key===row.model)&&!currentModel;
          return `<tr${currentModel?' class="ol-a-current-model"':''}><th scope="row">${canOpen?`<button class="ol-a-model-link" data-ol-a-model="${esc(row.model)}" type="button">${esc(modelLabel(row.model))}</button>`:esc(modelLabel(row.model))}${currentModel?'<span class="ol-a-current-label">Viewing</span>':''}</th><td>${number(row.rmse,4)}</td><td>${number(row.r2,3)}</td><td>${number(row.n,0)}</td></tr>`;
        }).join('')}</tbody></table>
      </div></section>`:''}
    <div class="ol-methods">
      <details><summary>${icon('info',17)} Where the comparison applies <span>${icon('down',17)}</span></summary><div>
        <p>The recorded dates were used as retrospective targets after model fitting and selection. They were reviewed during development; an unexamined later period is needed to confirm performance. Whole-lake results combine zones using weights from when the forecast was made.</p>
        <p>The scorecard and candidate ranking use all matched dates. The chart metrics use the visible date range. R² compares squared errors with predicting the mean observation; an R² below zero is worse than that benchmark. MAE averages the absolute differences, and RMSE gives more weight to large errors.</p>
      </div></details>
    </div>`;
    const showPlot=()=>{
      const title=el(content,'#ol-a-plot-title');
      title?.scrollIntoView({block:'start'});
      title?.focus({preventScroll:true});
    };
    const scoreButtons=content.querySelectorAll('[data-ol-cell]');
    scoreButtons.forEach(button=>button.addEventListener('click',()=>{
      current.indicator=button.dataset.indicator;
      current.days=+button.dataset.days;
      el(host,'#ol-a-indicator').value=current.indicator;
      el(host,'#ol-a-horizon').value=current.days;
      draw();
      showPlot();
    }));
    const dateInput=el(host,'#ol-a-date');
    dateInput.disabled=!scopedRows.length;
    dateInput.addEventListener('change',event=>{current.date=event.target.value;draw();el(host,'#ol-a-date')?.focus({preventScroll:true});});
    content.querySelectorAll('[data-ol-a-step]').forEach(button=>button.addEventListener('click',()=>{
      const next=scopedRows[selectedIndex + +button.dataset.olAStep];
      if(!next) return;
      current.date=datePart(next.target_date);
      const direction=button.dataset.olAStep;
      draw();
      const step=el(content,`[data-ol-a-step="${direction}"]`);
      (step?.disabled?el(content,'#ol-a-date'):step)?.focus({preventScroll:true});
    }));
    const svg=el(content,'[data-ol-a-plot] svg.ol-chart');
    if(svg && scopedRows.length>1) svg.addEventListener('click',event=>{
      const transform=svg.getScreenCTM();
      if(!transform) return;
      const pointer=new DOMPoint(event.clientX,event.clientY).matrixTransform(transform.inverse());
      if(pointer.x<63||pointer.x>785||pointer.y<25||pointer.y>236) return;
      const start=timestamp(scopedRows[0].target_date), end=timestamp(scopedRows.at(-1).target_date);
      const date=start+(end-start)*((pointer.x-63)/(785-63));
      const nearest=scopedRows.reduce((best,row)=>Math.abs(timestamp(row.target_date)-date)<Math.abs(timestamp(best.target_date)-date)?row:best);
      current.date=datePart(nearest.target_date);
      draw();
    });
    el(content,'[data-ol-a-range]').addEventListener('submit',event=>{
      event.preventDefault();
      const form=new FormData(event.currentTarget), from=form.get('from'), to=form.get('to');
      if(!validDate(from)||!validDate(to)||from>to||from<first||to>last) {
        el(content,'[data-ol-a-form-error]').textContent='Choose a start date on or before the end date, within the evaluation period.';
        return;
      }
      current.from=from;
      current.to=to;
      draw();
      showPlot();
    });
    content.querySelectorAll('[data-ol-a-reset]').forEach(button=>button.addEventListener('click',()=>{
      current.from=first;
      current.to=last;
      draw();
      showPlot();
    }));
    content.querySelectorAll('[data-ol-a-model]').forEach(button=>button.addEventListener('click',()=>{
      current.model=button.dataset.olAModel;
      el(host,'#ol-a-model').value=current.model;
      load({showChart:true});
    }));
    const exportButton=el(content,'[data-ol-download]');
    exportButton.disabled=!scopedRows.length;
    exportButton.addEventListener('click',()=>download(toCsv(scopedRows,['target_date','origin_date','actual_elapsed_days','horizon_days','zone_id','parameter','model','actual','prediction','error']),'aqualens-historical-pairs.csv','text/csv;charset=utf-8'));
  }
  async function load({showChart=false}={}) {
    const ticket=++turn, {model,zone}=current, key=`${model}:${zone}`;
    result=null;
    content.setAttribute('aria-busy','true');
    content.innerHTML=`<div class="ol-a-loading" role="status"><span class="ol-spinner" aria-hidden="true"></span><strong>Opening historical comparisons</strong><p>Loading exact prediction and observation pairs for the selected model and location.</p></div>`;
    try {
      const response=cache.get(key) || (mode==='snapshot'?(data.backtests?.[model] || (model==='det'?data.backtest:null)):
        await ctx.savedOutlook?.('backtest',{model,zone}) || await ctx.request(url('backtest',{dataset,model,zone}),{signal}));
      if(signal?.aborted||turn!==ticket) return;
      if(!response?.available || !Array.isArray(response.rows) || !response.rows.some(row=>String(row.zone_id)===zone)) {
        stateCard('No historical pairs available',dataset==='modis'?(response?.reason||'Choose another location or model to find a recorded comparison.'):'This dataset has no saved historical prediction pairs. Its forecast can still be used when a compatible model is available.',{related:action('Open forecast','forecasts')});
        return;
      }
      cache.set(key,response);
      result=response;
      draw();
      if(showChart) {
        const title=el(content,'#ol-a-plot-title');
        title?.scrollIntoView({block:'start'});
        title?.focus({preventScroll:true});
      }
    } catch(error) {
      if(!signal?.aborted&&ticket===turn) stateCard('Comparison unavailable',error.message || 'Unable to read the recorded evaluation.',{retry:true});
    } finally {
      if(!signal?.aborted&&ticket===turn) content.setAttribute('aria-busy','false');
    }
  }
  for(const [id,field] of Object.entries({'ol-a-zone':'zone','ol-a-model':'model','ol-a-horizon':'days','ol-a-indicator':'indicator'})) {
    el(host,`#${id}`).addEventListener('change',event=>{
      current[field]=field==='days'?+event.target.value:event.target.value;
      if(field==='model'||field==='zone') load();
      else draw();
    });
  }
  await load();
}
