import {escapeHtml as html, formatDate, formatNumber, icon, download, toCsv, safeUrl, parameterNames} from './utils.js';

const observationKeys = ['date','zone_id','cell_id','lat','lon','ndwi','turbidity_proxy','chlorophyll_proxy','valid_pixel_fraction'];
const number = (value, decimals = 0) => formatNumber(value, decimals);
const statusNames = {Critical:'Unusual', Warning:'Watch', Normal:'Stable'};

function pageIntro(kicker, title, description) {
  return `<header class="station-intro"><p class="station-kicker">${html(kicker)}</p><h1>${html(title)}</h1><p>${html(description)}</p></header>`;
}

function parameterOptions() {
  return '<option value="">All indicators</option>' + Object.entries(parameterNames).map(([id,label]) => `<option value="${id}">${html(label)}</option>`).join('');
}

function sourceCard(record) {
  const link = safeUrl(record.url);
  return `<article class="station-panel reference-card"><div class="reference-top"><span class="station-tag">${html(record.source_id || 'Reference')}</span>${record.date ? `<span>${html(formatDate(record.date))}</span>` : ''}</div>
    <h3>${html(record.title || 'Source')}</h3><p>${html(record.text || '')}</p>
    ${record.applicability ? `<div class="reference-note"><strong>How it applies</strong><p>${html(record.applicability)}</p></div>` : ''}
    ${link ? `<a class="station-text-link" href="${html(link)}" target="_blank" rel="noopener noreferrer">Open original source ${icon('external',15)}</a>` : ''}</article>`;
}

export function renderSources(host, ctx) {
  const collection = Array.isArray(ctx.data.knowledge) ? ctx.data.knowledge : [];
  host.innerHTML = pageIntro('SOURCE LIBRARY','Evidence behind the findings','Find the references used to interpret satellite signals and water use limits.')+
    `<section class="station-panel sources-tools">
      <form data-source-search class="station-fields">
        <label class="station-text-input"><span>Search sources</span><input type="search" name="query" placeholder="Try turbidity, runoff, or algae"></label>
        <label><span>Indicator</span><select name="parameter">${parameterOptions()}</select></label>
        <button type="submit" class="station-button primary">${icon('search',17)} Search</button>
      </form>
      <button type="button" data-add-sources class="station-button ghost"${ctx.mode === 'snapshot' ? ' disabled' : ''}>${icon('upload',17)} Import sources</button>
    </section><div class="station-section-header"><h2>Reference collection</h2><output data-reference-count aria-live="polite"></output></div><div class="reference-grid" data-source-results></div>`;
  const list = host.querySelector('[data-source-results]');
  const count = host.querySelector('[data-reference-count]');
  function fill(records) {
    count.textContent = `${records.length} ${records.length === 1 ? 'source' : 'sources'}`;
    list.innerHTML = records.length ? records.map(sourceCard).join('') : '<p class="station-empty">No matching references. Try fewer words or change the indicator.</p>';
  }
  fill(collection);
  let generation = 0;
  host.querySelector('[data-source-search]').onsubmit = async event => {
    event.preventDefault();
    const id = ++generation;
    const values = new FormData(event.currentTarget);
    const query = String(values.get('query') ?? '').trim(), parameter = values.get('parameter');
    if (!query && !parameter) { fill(collection); return; }
    if (ctx.mode === 'snapshot') {
      const tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
      fill(collection.filter(record => (!parameter || record.parameters?.includes(parameter) || !record.parameters?.length) &&
        tokens.every(token => `${record.title || ''} ${record.text || ''} ${record.applicability || ''}`.toLowerCase().includes(token))));
      return;
    }
    try {
      const args = new URLSearchParams({q:query, ...(parameter ? {parameter} : {}), ...(ctx.data.evidence_id ? {evidence_id:ctx.data.evidence_id} : {})});
      const response = await ctx.request(`/api/evidence?${args}`, {signal:ctx.signal});
      if (id === generation && !ctx.signal.aborted) fill(response.results || []);
    } catch (error) {
      if (!ctx.signal.aborted) ctx.notify(error.message, 'error');
    }
  };
  host.querySelector('[data-add-sources]').onclick = ctx.uploadSources;
}

function makeReportStats(report) {
  const coverage = report.coverage == null ? 'Unknown' : `${number(report.coverage * 100,1)}%`;
  return [['Samples',number(report.rows)], ['Observation dates',number(report.periods)], ['Location cells',number(report.cells)], ['Mean pixel coverage',coverage]];
}

function observationTable(rows) {
  const cols = [['date','Date'],['zone_id','Zone'],['cell_id','Cell'],['ndwi','NDWI'],['turbidity_proxy','Turbidity proxy'],['chlorophyll_proxy','Chlorophyll proxy']];
  return `<div class="station-table-scroll" role="region" aria-label="Data preview" tabindex="0"><table class="station-table"><thead><tr>${cols.map(([,label])=>`<th scope="col">${html(label)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${cols.map(([key], index)=>`<td>${html(index === 0 ? formatDate(row[key]) : index < 3 ? row[key] : number(row[key],3))}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

export function renderRecords(host, ctx) {
  const report = ctx.data.report || {};
  const rows = Array.isArray(ctx.data.observations) ? ctx.data.observations : [];
  const snapshot = ctx.mode === 'snapshot';
  const running = ['queued','running'].includes(ctx.state.job?.status);
  const trainReady = !snapshot && Number(report.periods) >= 35;
  host.innerHTML = pageIntro('WORKSPACE DATA','Your observations','Import a CSV, inspect the readings and quality notes, or run DET training on this dataset.') +
    `<section class="station-panel records-dataset"><div><p class="station-kicker">ACTIVE DATASET</p><h2>${html(report.filename || report.lake_id || 'Lake records')}</h2><p>${html(formatDate(report.start))} to ${html(formatDate(report.end))} · ${html(number(report.zones))} monitoring zones</p></div>
      <div class="station-action-row"><button type="button" class="station-button primary" data-upload${snapshot ? ' disabled' : ''}>${icon('upload',17)} Upload CSV</button>
      <button type="button" class="station-button ghost" data-export>${icon('download',17)} Export readings</button>
      ${ctx.data.dataset !== 'modis' ? '<button type="button" class="station-button ghost" data-use-supplied>Use supplied dataset</button>' : ''}</div>
    </section>
    <dl class="records-stats">${makeReportStats(report).map(([label,value])=>`<div><dt>${html(label)}</dt><dd>${html(value)}</dd></div>`).join('')}</dl>
    <div class="records-grid"><section class="station-panel"><h2>Input checks</h2><dl class="station-definitions">
      <div><dt>Rows removed as invalid</dt><dd>${html(number(report.invalid_rows))}</dd></div>
      <div><dt>Duplicate rows removed</dt><dd>${html(number(report.duplicate_rows))}</dd></div>
      <div><dt>NDWI values outside the valid range</dt><dd>${html(number(report.ndwi_invalid_count))}</dd></div></dl>
      ${(report.warnings || []).length ? `<details class="station-details"><summary>Read the dataset notes</summary><ul>${report.warnings.map(item=>`<li>${html(item)}</li>`).join('')}</ul></details>` : ''}</section>
    <section class="station-panel"><h2>Train DET</h2><p>The model uses earlier observations for training and later dates to check accuracy. Training requires at least 35 observation dates.</p>
      ${snapshot ? '<p class="station-callout">Start the local engine to train on a dataset.</p>' : ''}
      <form class="station-fields" data-training-form><label><span>Maximum epochs</span><input type="number" name="epochs" min="1" max="300" value="120" required></label><button type="submit" class="station-button primary"${!trainReady || running ? ' disabled' : ''}>Start training</button></form>
      <div data-training-status class="station-training" ${ctx.state.job ? '' : 'hidden'}><strong>${html(ctx.state.job?.status === 'complete' ? 'Training complete' : ctx.state.job?.status === 'failed' ? 'Training failed' : 'Training in progress')}</strong><progress max="100" value="${Number(ctx.state.job?.progress) || 0}" aria-label="Training progress"></progress><p>${html(ctx.state.job?.error || ctx.state.job?.message || '')}</p></div>
    </section></div>
    <section class="station-panel records-preview">
      <div class="station-section-header"><div><h2>Browse recorded values</h2><p>Proxy estimates use the scale supplied in your CSV.</p></div><label class="station-search-label"><span>Filter by date or zone</span><input type="search" data-observation-search placeholder="2023-02 or Z1"></label></div>
      <div data-rows></div>
      <div class="station-pager"><button class="station-button ghost" type="button" data-back>Previous</button><output data-page-count aria-live="polite"></output><button class="station-button ghost" type="button" data-more>Next</button></div>
    </section>
    <details class="station-panel station-details"><summary>What columns should my CSV include?</summary><p>Use <code>date, lat, lon, ndwi, turbidity_proxy, chlorophyll_proxy</code>. The original Earth Engine export using <code>system:index</code> and <code>.geo</code> also works. Add <code>zone_id</code>, <code>cell_id</code>, <code>is_observed</code>, and <code>valid_pixel_fraction</code> when available.</p></details>`;
  const pageSize = 20, recentFirst = [...rows].reverse();
  let page = 0, filtered = recentFirst;
  const back = host.querySelector('[data-back]'), more = host.querySelector('[data-more]');
  function refreshPage() {
    const first = page * pageSize;
    const pageRows = filtered.slice(first, first + pageSize);
    host.querySelector('[data-rows]').innerHTML = pageRows.length ? observationTable(pageRows) : '<p class="station-empty">No matching observations.</p>';
    host.querySelector('[data-page-count]').textContent = pageRows.length ? `${number(first+1)}–${number(first+pageRows.length)} of ${number(filtered.length)}` : '0 observations';
    back.disabled = !page;
    more.disabled = first + pageSize >= filtered.length;
  }
  refreshPage();
  back.onclick = () => { if (page > 0) { page -= 1; refreshPage(); } };
  more.onclick = () => { if ((page + 1) * pageSize < filtered.length) { page += 1; refreshPage(); } };
  host.querySelector('[data-observation-search]').oninput = event => {
    const query = event.target.value.trim().toLowerCase();
    filtered = query ? recentFirst.filter(row=>`${row.date} ${row.zone_id} ${row.zone_name} ${row.cell_id}`.toLowerCase().includes(query)) : recentFirst;
    page = 0; refreshPage();
  };
  host.querySelector('[data-training-form]').onsubmit = event => {
    event.preventDefault();
    if (trainReady) ctx.train(+new FormData(event.currentTarget).get('epochs'));
  };
  host.querySelector('[data-upload]').onclick = ctx.upload;
  host.querySelector('[data-export]').onclick = () => download(toCsv(rows, observationKeys), 'aqualens-observations.csv', 'text/csv;charset=utf-8');
  const bundled = host.querySelector('[data-use-supplied]');
  if (bundled) bundled.onclick = ctx.useBundled;
}

export function renderAlerts(host, ctx) {
  const alerts = Array.isArray(ctx.data.alerts) ? ctx.data.alerts : [];
  const statuses = record => record.status || statusNames[record.severity] || record.severity;
  host.innerHTML = pageIntro('MONITORING','Changes to review','Compare the latest signal in each zone with its recent history. An alert flags a change for investigation.')+
    `<div class="station-toolbar"><label><span>Alert filter</span><select data-alert-filter><option value="flagged">Watch and Unusual</option><option value="all">All signals</option><option>Watch</option><option>Unusual</option><option>Insufficient data</option></select></label><a href="#assessment" class="station-button ghost">Back to water condition</a></div>
    <div class="reference-grid" data-alerts></div>`;
  function show(filter) {
    const filtered = alerts.filter(record => filter === 'all' || (filter === 'flagged' ? ['Watch','Unusual'].includes(statuses(record)) : statuses(record) === filter));
    host.querySelector('[data-alerts]').innerHTML = filtered.length ? filtered.map(record => `<article class="station-panel alert-entry">
      <div class="reference-top"><span class="station-tag" data-tone="${['Watch','Unusual'].includes(statuses(record)) ? 'notice' : 'plain'}">${html(statuses(record))}</span><span>${html(formatDate(record.date))}</span></div>
      <h3>${html(parameterNames[record.parameter] || record.parameter)}</h3><p><strong>${html(record.zone_name || record.zone_id)}</strong> · ${html(number(record.value,3))}</p>
      <p>${html(record.reason || '')}</p><button type="button" class="station-button ghost" data-view-zone="${html(record.zone_id)}">Explore this zone</button>
    </article>`).join('') : '<p class="station-empty">No signals match this filter.</p>';
    host.querySelectorAll('[data-view-zone]').forEach(button => { button.onclick = () => { ctx.state.targetZone = button.dataset.viewZone; ctx.navigate('zones'); }; });
  }
  show('flagged');
  host.querySelector('[data-alert-filter]').onchange = event => show(event.target.value);
}
