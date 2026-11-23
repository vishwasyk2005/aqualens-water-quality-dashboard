import {escapeHtml as html, formatNumber, formatDate, icon, download, safeUrl} from './utils.js';

const list = value => Array.isArray(value) ? value : [];
const statusTone = value => ({stable:'stable', typical:'stable', normal:'stable', watch:'caution', unusual:'caution', elevated:'caution', low:'caution'})[String(value).toLowerCase()] || 'muted';
const statusText = value => ({stable:'Within recent range', watch:'Watch', unusual:'Unusual', 'insufficient data':'Not enough data'})[String(value).toLowerCase()] || value || 'Not enough data';
const useIcons = {drinking:'drop', recreation:'waves', irrigation:'layers', aquatic_life:'activity'};
const parameterIcons = {ndwi:'waves', turbidity_proxy:'layers', chlorophyll_proxy:'activity', lake_surface_temp_c:'activity', tsm_mg_l:'layers'};
const useNames = {drinking:'Drinking', recreation:'Bathing & recreation', irrigation:'Irrigation', aquatic_life:'Fish habitat'};
const isNumber = value => value != null && value !== '' && Number.isFinite(+value);
const datePart = value => /^\d{4}-\d{2}-\d{2}/.test(String(value)) ? String(value).slice(0,10) : '';
const readableDate = value => datePart(value) ? `<time datetime="${html(datePart(value))}">${html(formatDate(value))}</time>` : 'Date unavailable';
const percentage = value => `${formatNumber(+value * 100, 0)}%`;

function citationLinks(ids, sources) {
  const indexes = [...new Set(list(ids).map(id => sources.findIndex(source => source.source_id === id)))].filter(index => index >= 0);
  if (!indexes.length) return '';
  return `<span class="quality-citations" aria-label="Supporting sources">${indexes.map(index => `<button type="button" data-q-cite="${index}" aria-label="Read source ${index + 1}: ${html(sources[index].title)}">${index + 1}</button>`).join('')}</span>`;
}

function readingCard(reading, signal, sources) {
  const delta = reading.change;
  const comparable = isNumber(delta);
  const difference = comparable ? `${+delta > 0 ? '+' : ''}${formatNumber(delta, 3)}` : null;
  const direction = comparable ? +delta > 0 ? 'Higher' : +delta < 0 ? 'Lower' : 'Unchanged' : null;
  const visibleStatus = reading.status_label || statusText(reading.status);
  const interpretation = reading.interpretation || signal?.interpretation || reading.reason;
  const sameRange = isNumber(reading.range_low) && isNumber(reading.range_high);
  const count = isNumber(reading.sample_count) ? `${formatNumber(reading.sample_count, 0)} usable${isNumber(reading.reported_count) ? ` / ${formatNumber(reading.reported_count, 0)} reported` : ''}` : '';
  const coverage = isNumber(reading.coverage) ? `${percentage(reading.coverage)} valid pixels` : reading.coverage_unknown ? 'Pixel quality unavailable' : '';
  return `<article class="quality-reading">
    <div class="quality-reading-head"><span class="quality-symbol">${icon(parameterIcons[reading.parameter] || 'activity',18)}</span><span class="quality-badge ${statusTone(reading.status)}">${html(visibleStatus)}</span></div>
    <h3>${html(reading.label || reading.parameter)}</h3>
    <div class="quality-number">${formatNumber(reading.value, 3)}</div>
    <p class="quality-unit">${html(reading.unit || reading.units || 'Supplied scale')}${datePart(reading.date) ? ` · ${readableDate(reading.date)}` : ''}</p>
    <div class="quality-difference">${difference && reading.previous_date ? `<strong>${html(direction ? `${direction} ${difference}` : difference)}</strong><span>since ${readableDate(reading.previous_date)}</span>` : '<span>No usable previous reading</span>'}</div>
    ${interpretation ? `<p class="quality-meaning">${html(interpretation)}</p>` : ''}
    ${sameRange || count || coverage || reading.reason && interpretation !== reading.reason ? `<details class="quality-disclosure"><summary>Reading details</summary>
      ${sameRange ? `<p><strong>Recent comparison band</strong> ${formatNumber(reading.range_low,3)} to ${formatNumber(reading.range_high,3)}${isNumber(reading.history_periods) ? ` using ${formatNumber(reading.history_periods,0)} prior observation dates` : ''}.</p>` : ''}
      ${count || coverage ? `<p>${html([count, coverage].filter(Boolean).join(' · '))}</p>` : ''}
      ${reading.reason && interpretation !== reading.reason ? `<p class="quality-note">${html(reading.reason)}</p>` : ''}
    </details>` : ''}
    ${citationLinks(reading.evidence_ids || signal?.evidence_ids, sources)}
  </article>`;
}

function observationSource(monitoring, report) {
  const source = monitoring?.source;
  if (typeof source === 'string' && source) return source;
  return source?.label || source?.sensor || report.provenance?.sensor_filter || 'Uploaded satellite observations';
}

function satelliteFinding(monitoring, report, readings) {
  const statusCode = monitoring?.status?.code || monitoring?.status || report.status?.code;
  const tone = statusTone(statusCode);
  const observed = monitoring?.date || report.as_of;
  const rowCount = monitoring?.reported_count ?? monitoring?.source?.reported_count ?? report.provenance?.source_rows;
  const coreReadings = readings.filter(reading => !datePart(reading.date) || datePart(reading.date) === datePart(observed));
  const hasCoverage = coreReadings.some(reading => isNumber(reading.coverage));
  const hasUnknownCoverage = coreReadings.some(reading => reading.coverage_unknown);
  const dateLabel = coreReadings.length !== readings.length ? 'Optical observation date' : 'Latest saved observation';
  const headline = monitoring?.headline || report.status?.label || 'No usable satellite finding';
  const detail = monitoring?.summary || report.preliminary?.title || report.status?.summary;
  const priority = list(monitoring?.priority_zones);
  return `<div class="quality-overview">
    <article class="quality-finding" data-tone="${tone}"><div class="quality-finding-lead">
      <span class="quality-symbol">${icon(tone==='caution' ? 'alert' : 'activity',23)}</span><span class="quality-label">Observed surface condition</span>
    </div><h3>${html(headline)}</h3>${detail ? `<p>${html(detail)}</p>` : ''}
    ${priority.length ? `<div class="quality-priority"><span>Review on map</span>${priority.slice(0,4).map(zone => {
      const name = typeof zone === 'string' ? zone : zone.zone_name || zone.label || zone.zone_id;
      const id = typeof zone === 'string' ? null : zone.zone_id;
      return id ? `<button type="button" data-q-map data-q-zone="${html(id)}">${html(name)}</button>` : `<strong>${html(name)}</strong>`;
    }).join('')}</div>` : ''}
    <div class="quality-finding-actions"><button class="station-button" type="button" data-q-map>${icon('map',17)} Inspect observations</button><button class="station-button lime" type="button" data-q-forecast>${icon('chart',17)} View forecast</button></div>
    </article>
    <aside class="quality-observation" aria-label="Satellite observation details"><span class="quality-label">${icon('calendar',16)} ${dateLabel}</span><strong>${readableDate(observed)}</strong>
      <dl><dt>Source</dt><dd>${html(observationSource(monitoring,report))}</dd>
      ${isNumber(rowCount) ? `<dt>Reported samples</dt><dd>${formatNumber(rowCount,0)}</dd>` : ''}
      <dt>Pixel quality</dt><dd>${hasCoverage ? hasUnknownCoverage ? 'Partly reported' : 'See indicator details' : 'Not supplied'}</dd></dl>
      ${monitoring?.source?.stale ? '<p class="quality-date-note">Historical reading. A newer satellite CSV can update this report.</p><button class="station-button" type="button" data-q-upload>Upload newer readings</button>' : datePart(observed) ? '<p>Conditions may have changed after this observation.</p>' : ''}
    </aside>
  </div>`;
}

function followUp(monitoring, report, sources) {
  const readings = list(monitoring?.indicators);
  const flagged = readings.filter(reading => statusTone(reading.status) === 'caution' && reading.action);
  if (flagged.length) {
    return flagged.map(reading => `<article class="quality-precaution">
      <span class="quality-symbol">${icon(parameterIcons[reading.parameter] || 'target',18)}</span>
      <div><div class="quality-precaution-title"><h3>${html(reading.label)}</h3></div><p>${html(reading.action)}</p>
      ${citationLinks(reading.evidence_ids, sources)}</div>
    </article>`).join('');
  }
  const priorities = list(monitoring?.priority_zones);
  if (priorities.length) {
    return priorities.slice(0,4).map(zone => `<article class="quality-precaution">
      <span class="quality-symbol">${icon('map',18)}</span>
      <div><h3>${html(zone.zone_name || zone.zone_id)}</h3>
      ${list(zone.indicators).length ? `<p>${html(zone.indicators.join(', '))}</p>` : ''}
      <p>${html(zone.reason || 'Compare this zone with the next clear satellite image.')}</p>
      <button type="button" class="station-button" data-q-map data-q-zone="${html(zone.zone_id)}">Inspect zone on map</button></div></article>`).join('');
  }
  if (monitoring) {
    return `<article class="quality-precaution"><span class="quality-symbol">${icon('target',18)}</span>
      <div><h3>Compare the next usable satellite observation</h3><p>Compare the same locations and check the image-quality flags where they are available.</p></div>
    </article>`;
  }
  const optical = list(report.precautions).filter(item => item.id === 'check-optical');
  if (optical.length) return optical.map(item => precautionCard(item, sources)).join('');
  return `<article class="quality-precaution"><span class="quality-symbol">${icon('target',18)}</span>
    <div><div class="quality-precaution-title"><h3>Compare the next usable observation</h3></div><p>Review the same locations on the map and check whether the surface readings change. Inspect pixel quality where it is available.</p></div>
  </article>`;
}

function outlookMarkup(result) {
  const outlook = result?.projection;
  const horizons = list(outlook?.horizons);
  if (!horizons.length) {
    return `<p class="quality-outlook-note">${html(result?.forecast_error || 'There is no forecast for this location and dataset yet.')}</p>`;
  }
  const archived = horizons.every(row => Date.parse(`${datePart(row.target_date)}T23:59:59`) < Date.now());
  return `<p class="quality-outlook-note">${archived ? 'Archive outlook from' : 'Forecast origin'} ${readableDate(outlook.origin_date)}. Open a lead time to review the estimates and their accuracy.</p>
    <div class="quality-outlook-grid">${horizons.map(horizon => `<button type="button" class="quality-outlook-item" data-q-forecast data-q-horizon="${html(horizon.horizon_days)}">
      <span class="quality-outlook-target"><strong>${html(horizon.horizon_days)} days after the reading</strong><span>${readableDate(horizon.target_date)}</span></span>
      <span class="quality-outlook-headline" data-tone="${statusTone(horizon.status)}">${html(horizon.headline || statusText(horizon.status))}</span>
      ${horizon.summary ? `<small>${html(horizon.summary)}</small>` : ''}
    </button>`).join('')}</div>`;
}

function usePanel(use, index, sources) {
  if (!use) return '<p class="quality-empty">No guidance is available for this report.</p>';
  const prelim = use.preliminary;
  const findings = list(prelim?.findings).filter(finding => typeof finding === 'string' && finding.trim());
  const tests = list(use.missing_tests);
  const label = use.status === 'not_established' ? 'Use checks pending' : use.label || 'Use checks pending';
  return `<article class="quality-use-panel" id="quality-use-panel" aria-labelledby="quality-use-tab-${index}" role="tabpanel" tabindex="0">
    <div class="quality-use-action">
      <h3>${html(prelim?.label || use.title)}</h3>
      <span class="quality-badge caution">${html(label)}</span>
      ${findings.length ? `<ul class="quality-use-findings">${findings.map(finding => `<li>${html(finding)}</li>`).join('')}</ul>` : prelim?.summary ? `<p class="quality-use-fallback">${html(prelim.summary)}</p>` : ''}
      ${prelim?.action ? `<div class="quality-next-step"><span>${icon('target',17)} What to do next</span><p>${html(prelim.action)}</p></div>` : ''}
      ${citationLinks(prelim?.evidence_ids, sources)}
    </div>
    <details class="quality-use-tests quality-disclosure">
      <summary>How to verify this use</summary>
      ${use.summary ? `<p>${html(use.summary)}</p>` : ''}
      ${use.action ? `<p>${html(use.action)}</p>` : ''}
      ${tests.length ? `<ul>${tests.map(test => `<li>${html(typeof test === 'object' ? test?.label || test?.name || test?.parameter : test)}</li>`).join('')}</ul>` : ''}
      ${citationLinks(use.evidence_ids, sources)}
    </details>
  </article>`;
}

function sourceRows(sources) {
  return sources.map((source, index) => {
    const url = safeUrl(source.url);
    const domain = url ? new URL(url).hostname.replace(/^www\./, '') : 'Source URL unavailable';
    return `<details class="quality-source" data-q-source="${index}">
      <summary><span class="quality-reference-no">${String(index + 1).padStart(2, '0')}</span><span><strong>${html(source.title || 'Untitled source')}</strong><small>${html(domain)}${source.verified_on ? ` · Reviewed ${html(formatDate(source.verified_on))}` : ''}</small></span><span class="quality-expand" aria-hidden="true">+</span></summary>
      <div class="quality-passage">
        <blockquote>${html(source.text || 'No saved extract is available.')}</blockquote>
        ${source.applicability ? `<p><strong>Applies to:</strong> ${html(source.applicability)}</p>` : ''}
        ${url ? `<a class="station-text-link" target="_blank" rel="noopener noreferrer" href="${html(url)}">Read the original source</a>` : ''}
      </div>
    </details>`;
  }).join('');
}

function precautionCard(precaution, sources) {
  const priority = ({high:'Before use', conditional:'Conditional', routine:'Follow-up'})[precaution.priority] || 'Field check';
  return `<article class="quality-precaution"><span class="quality-symbol">${icon(precaution.priority === 'routine' ? 'target' : 'alert',18)}</span>
    <div><div class="quality-precaution-title"><h3>${html(precaution.title)}</h3><span class="quality-badge ${precaution.priority === 'routine' ? 'muted' : 'caution'}">${html(priority)}</span></div>
    ${precaution.trigger ? `<p class="quality-note"><strong>When:</strong> ${html(precaution.trigger)}</p>` : ''}<p>${html(precaution.action)}</p>
    <details class="quality-disclosure"><summary>Why this check matters</summary><p>${html(precaution.why)}</p><p class="quality-note">${html(precaution.evidence_note)}</p></details>
    ${citationLinks(precaution.evidence_ids, sources)}</div>
  </article>`;
}

export function renderQuality(host, ctx) {
  const reports = list(ctx.data.water_assessments);
  if (!reports.length) {
    host.innerHTML = '<section class="quality-report"><p class="quality-empty">No water assessment is available for this dataset.</p></section>';
    return;
  }
  const monitorFor = report => report.monitoring || ctx.data.satellite_monitoring?.[report.zone_id] ||
    list(ctx.data.satellite_monitoring).find(item => String(item.zone_id) === String(report.zone_id));
  const sourcesFor = report => [...new Map([...list(report.sources), ...list(monitorFor(report)?.sources)].map(source => [source.source_id, source])).values()];
  const initial = reports.find(report => String(report.zone_id) === String(ctx.state.qualityZone || 'all')) || reports[0];
  let report = initial;
  let activeIndex = 0;
  let outlookRequest;
  const projections = new Map();
  const stopOutlook = () => outlookRequest?.abort();
  ctx.signal?.addEventListener('abort', stopOutlook, {once:true});

  async function loadOutlook(zone) {
    const mount = host.querySelector('[data-q-outlook]');
    if (!mount || ctx.signal?.aborted) return;
    const saved = projections.get(zone);
    if (saved) {
      mount.innerHTML = outlookMarkup(saved);
      return;
    }
    if (ctx.mode !== 'api') {
      const projection = ctx.data.monitoring_projections?.auto?.[zone];
      mount.innerHTML = projection ? outlookMarkup({projection}) : '<p class="quality-outlook-note">Open Forecast to review projections saved with this dataset.</p>';
      return;
    }
    const controller = new AbortController();
    outlookRequest = controller;
    const query = new URLSearchParams({dataset:ctx.data.dataset || 'modis', zone, model:'auto'});
    if (ctx.data.evidence_id) query.set('evidence_id',ctx.data.evidence_id);
    try {
      const result = await ctx.request(`/api/monitoring?${query}`, {signal:controller.signal});
      if (controller.signal.aborted || ctx.signal?.aborted || !mount.isConnected) return;
      projections.set(zone,result);
      mount.innerHTML = outlookMarkup(result);
    } catch (error) {
      if (!controller.signal.aborted && !ctx.signal?.aborted && mount.isConnected) {
        mount.innerHTML = '<p class="quality-outlook-note">The condition outlook could not be loaded. Open Forecast to review model availability.</p>';
      }
    } finally {
      if (outlookRequest === controller) outlookRequest = undefined;
    }
  }

  function update(useId, options = {}) {
    stopOutlook();
    const monitoring = monitorFor(report);
    const sources = sourcesFor(report);
    const uses = list(report.uses).filter(use => use.id !== 'monitoring');
    const readingRows = monitoring ? list(monitoring.indicators) : list(report.observations);
    const signals = list(report.preliminary?.signals);
    const usePrecautions = list(report.precautions).filter(item => item.id !== 'check-optical');
    const confidence = report.preliminary?.confidence;
    activeIndex = Math.max(0, uses.findIndex(use => use.id === useId));
    const usesOpen = host.querySelector('.quality-use')?.open ?? ctx.state.qualityUsesOpen ?? true;
    host.innerHTML = `<section class="quality-report" aria-labelledby="quality-heading">
      <header class="quality-header"><div>
        <div class="quality-eyebrow">${icon('activity',16)} Satellite monitoring</div>
        <h1 id="quality-heading">Lake condition</h1>
      </div><div class="quality-controls">
        <label for="quality-location"><span>Monitoring location</span><select id="quality-location">
          ${reports.map(zone => `<option value="${html(zone.zone_id)}"${zone.zone_id === report.zone_id ? ' selected' : ''}>${html(zone.zone_name || zone.zone_id)}</option>`).join('')}
        </select></label>
        <button class="station-button" type="button" data-q-download>${icon('download',17)} Export condition</button>
      </div></header>
      ${satelliteFinding(monitoring, report, readingRows)}
      <section class="quality-outlook" aria-labelledby="quality-outlook-heading">
        <header class="quality-subheader"><h2 id="quality-outlook-heading">Projection after the last reading</h2></header>
        <div data-q-outlook aria-live="polite"><p class="quality-outlook-note">Checking the available forecast…</p></div>
      </section>
      <section class="quality-readings" aria-labelledby="quality-readings-heading">
        <header class="quality-subheader"><div><h2 id="quality-readings-heading">What changed in the readings</h2></div></header>
        ${readingRows.length ? `<div class="quality-grid">${readingRows.map(reading => readingCard(reading, signals.find(signal => signal.parameter === reading.parameter), sources)).join('')}</div>` : '<p class="quality-empty">No readings are available for this location.</p>'}
        ${confidence ? `<details class="quality-confidence quality-disclosure"><summary>Interpretation confidence: ${html(confidence.label)}</summary>
          <p>${html(confidence.meaning)}</p>${list(confidence.basis).length ? `<ul>${confidence.basis.map(reason => `<li>${html(reason)}</li>`).join('')}</ul>` : ''}
          ${report.preliminary?.scope_note ? `<p>${html(report.preliminary.scope_note)}</p>` : ''}
        </details>` : ''}
      </section>
      ${uses.length ? `<details class="quality-use"${usesOpen ? ' open' : ''}>
        <summary id="quality-use-heading">Water-use guidance and precautions</summary>
        <div class="quality-use-content">
          <div class="quality-tabs" role="tablist" aria-label="Water uses">${uses.map((use, index) => `<button type="button" id="quality-use-tab-${index}" data-q-use="${index}" aria-controls="quality-use-panel" aria-selected="${index === activeIndex}" role="tab" tabindex="${index === activeIndex ? 0 : -1}">${icon(useIcons[use.id] || 'drop',18)}${html(useNames[use.id] || use.title)}</button>`).join('')}</div>
          <div data-q-use-content>${usePanel(uses[activeIndex], activeIndex, sources)}</div>
          ${usePrecautions.length ? `<div class="quality-use-precautions">${usePrecautions.map(item => precautionCard(item, sources)).join('')}</div>` : ''}
        </div></details>` : ''}
      <section class="quality-evidence" aria-label="Precautions and evidence">
        <section class="quality-checks" aria-labelledby="quality-checks-heading">
          <header class="quality-subheader"><div><h2 id="quality-checks-heading">What to monitor next</h2></div></header>
          ${followUp(monitoring, report, sources)}
        </section>
        <section class="quality-references" aria-labelledby="quality-references-heading">
          <header class="quality-subheader"><div><h2 id="quality-references-heading">Supporting evidence</h2><p>References explain how to interpret the recorded readings.</p></div></header>
          ${sources.length ? sourceRows(sources) : '<p class="quality-empty">No matching references were retrieved for this report.</p>'}
          ${list(report.limitations).length ? `<details class="quality-confidence quality-disclosure"><summary>Limits of this assessment</summary><ul>${report.limitations.map(item => `<li>${html(item)}</li>`).join('')}</ul></details>` : ''}
        </section>
      </section>
    </section>`;
    const useDisclosure = host.querySelector('.quality-use');
    if (useDisclosure) useDisclosure.ontoggle = () => {
      if (useDisclosure.isConnected) ctx.state.qualityUsesOpen = useDisclosure.open;
    };
    void loadOutlook(report.zone_id);
    if (options.focusLocation) host.querySelector('#quality-location')?.focus({preventScroll:true});
  }

  function activate(index, focus) {
    const uses = list(report.uses).filter(use => use.id !== 'monitoring');
    if (!uses[index]) return;
    activeIndex = index;
    ctx.state.qualityUse = uses[index].id;
    host.querySelectorAll('[data-q-use]').forEach((tab, position) => {
      tab.setAttribute('aria-selected', String(position === index));
      tab.tabIndex = position === index ? 0 : -1;
      if (focus && position === index) tab.focus();
    });
    const content = host.querySelector('[data-q-use-content]');
    if (content) content.innerHTML = usePanel(uses[index], index, sourcesFor(report));
  }

  host.onchange = event => {
    if (event.target.id !== 'quality-location') return;
    const selected = reports.find(item => String(item.zone_id) === event.target.value);
    if (!selected) return;
    report = selected;
    ctx.state.qualityZone = selected.zone_id;
    update(ctx.state.qualityUse, {focusLocation:true});
  };
  host.onclick = event => {
    const control = event.target.closest('button');
    if (!control || !host.contains(control)) return;
    if (control.hasAttribute('data-q-use')) activate(Number(control.dataset.qUse), false);
    if (control.hasAttribute('data-q-upload')) ctx.upload();
    if (control.hasAttribute('data-q-map')) {
      const monitoring = monitorFor(report);
      const selected = list(monitoring?.indicators).find(item => ['ndwi','turbidity_proxy','chlorophyll_proxy'].includes(item.parameter) && statusTone(item.status) === 'caution');
      ctx.state.targetZone = control.dataset.qZone || report.zone_id;
      ctx.state.geography = {...ctx.state.geography, dataset:String(ctx.data.dataset ?? ''), date:datePart(monitoring?.date || report.as_of), parameter:selected?.parameter || 'ndwi'};
      ctx.navigate('map');
    }
    if (control.hasAttribute('data-q-forecast')) {
      const monitor = monitorFor(report);
      const selected = list(monitor?.indicators).find(item => ['ndwi','turbidity_proxy','chlorophyll_proxy'].includes(item.parameter) && statusTone(item.status) === 'caution');
      const days = Number(control.dataset.qHorizon);
      const pinned = [8,16,32].includes(days) ? {days, offset:days, model:'auto'} : {};
      ctx.state.forecast = {...ctx.state.forecast, zone:report.zone_id, ...(selected ? {indicator:selected.parameter} : {}), ...pinned};
      ctx.navigate('forecasts');
    }
    if (control.hasAttribute('data-q-download')) {
      const scope = String(report.zone_id || 'lake').replace(/[^a-z0-9_-]/gi, '_');
      const day = String(report.as_of || 'undated').slice(0, 10).replace(/[^a-z0-9-]/gi, '');
      download(JSON.stringify({...report, monitoring:monitorFor(report) || undefined}, null, 2), `aqualens-assessment-${scope}-${day}.json`, 'application/json;charset=utf-8');
      ctx.notify('Lake condition exported with its linked sources.');
    }
    if (control.hasAttribute('data-q-cite')) {
      const index = Number(control.dataset.qCite);
      if (!Number.isInteger(index) || index < 0) return;
      const source = host.querySelector(`[data-q-source="${index}"]`);
      if (!source) return;
      source.open = true;
      source.querySelector('summary')?.focus({preventScroll:true});
      source.scrollIntoView({block:'start', behavior:matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'});
    }
  };
  host.onkeydown = event => {
    if (!event.target.closest('[data-q-use]') || !['ArrowRight','ArrowLeft','Home','End'].includes(event.key)) return;
    const count = list(report.uses).filter(use => use.id !== 'monitoring').length;
    if (!count) return;
    event.preventDefault();
    const index = event.key === 'Home' ? 0 : event.key === 'End' ? count - 1 : (activeIndex + count + (event.key === 'ArrowRight' ? 1 : -1)) % count;
    activate(index, true);
  };
  update(ctx.state.qualityUse);
  return () => {
    stopOutlook();
    ctx.signal?.removeEventListener('abort', stopOutlook);
    host.onclick = host.onchange = host.onkeydown = null;
  };
}
