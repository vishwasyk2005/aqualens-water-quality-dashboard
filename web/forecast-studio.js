/* Presentation-only forecast studio. Scientific values are supplied by app.js. */
(() => {
  'use strict';
  const escape = value => String(value ?? '').replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
  const dateLabel = value => {
    const date = new Date(String(value ?? '').slice(0, 10) + 'T12:00:00');
    return Number.isFinite(+date) ? date.toLocaleDateString('en-GB', {day:'numeric', month:'short', year:'numeric'}) : 'Not available';
  };
  const numeric = value => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));
  const print = (value, precision = 3) => numeric(value) ? Number(value).toFixed(precision) : '—';
  const day = value => /^\d{4}-\d{2}-\d{2}$/.test(String(value)) ? Date.parse(value) : NaN;
  const cleanPoints = rows => (rows || []).filter(row => numeric(row.value) && Number.isFinite(day(row.date))).slice().sort((first, second) => first.date.localeCompare(second.date));

  function plot({history, future, latestDate, selectedDate, modelName = 'DET', showBand = true, validatedDays = 32} = {}) {
    const observed = cleanPoints(history), predicted = cleanPoints(future), intervals = showBand ? predicted.filter(row => numeric(row.lower) && numeric(row.upper)) : [];
    const points = observed.concat(predicted), values = points.map(row => +row.value).concat(intervals.flatMap(row => [+row.lower, +row.upper]));
    if (!points.length) return '<div class="empty-state">No usable observations or estimates for this selection.</div>';
    const left = 66, right = 860, top = 32, bottom = 255, begin = Math.min(...points.map(row => day(row.date))), end = Math.max(...points.map(row => day(row.date)));
    const first = Math.min(...values), last = Math.max(...values), padding = (last - first) * .13 || Math.max(.01, Math.abs(last) * .15), low = first - padding, high = last + padding;
    const x = time => left + (time - begin) / (end - begin || 1) * (right - left), y = value => bottom - (value - low) / (high - low) * (bottom - top);
    const anchor = day(latestDate), divider = Math.max(left, Math.min(right, x(Number.isFinite(anchor) ? anchor : end)));
    const extended = Number.isFinite(anchor) && anchor + validatedDays * 86400000 < end, extension = extended ? Math.max(divider, x(anchor + validatedDays * 86400000)) : right;
    const path = (rows, key, maxGap = 18) => rows.map((row, index) => `${index > 0 && day(row.date) - day(rows[index - 1].date) <= maxGap * 86400000 ? 'L' : 'M'}${x(day(row.date)).toFixed(2)},${y(+row[key]).toFixed(2)}`).join(' ');
    const ticks = Array.from({length:5}, (_, index) => {
      const level = low + (high - low) * index / 4, yy = y(level);
      return `<line class="lab-plot-grid" x1="${left}" y1="${yy}" x2="${right}" y2="${yy}"/><text x="${left - 9}" y="${yy + 4}" text-anchor="end">${escape(print(level, Math.abs(high - low) < .12 ? 3 : 2))}</text>`;
    }).join('');
    const dates = [...new Set([0,.25,.5,.75,1].map(fraction => Math.round((begin + (end - begin) * fraction) / 86400000) * 86400000))];
    const timeTicks = dates.map((time, index) => `<text x="${x(time)}" y="285" text-anchor="${index === 0 ? 'start' : index === dates.length - 1 ? 'end' : 'middle'}">${escape(new Date(time).toLocaleDateString('en-GB',{day:'numeric',month:'short',year:'2-digit',timeZone:'UTC'}))}</text>`).join('');
    const contiguous = [];
    intervals.forEach(row => {
      const sequence = contiguous[contiguous.length - 1];
      if (!sequence || day(row.date) - day(sequence[sequence.length - 1].date) > 86400000) contiguous.push([row]);
      else sequence.push(row);
    });
    const bands = contiguous.map(group => group.length > 1 ? `<path class="lab-plot-band" d="${path(group, 'upper')} ${group.slice().reverse().map(row => `L${x(day(row.date)).toFixed(2)},${y(+row.lower).toFixed(2)}`).join(' ')} Z"/>` : `<line class="lab-plot-band-single" x1="${x(day(group[0].date))}" x2="${x(day(group[0].date))}" y1="${y(+group[0].lower)}" y2="${y(+group[0].upper)}"/>`).join('');
    const availableAtCutoff = observed.at(-1)?.date === latestDate && predicted.length && day(predicted[0].date) - anchor <= 86400000;
    const selected = predicted.find(row => row.date === selectedDate), cursorX = selected ? x(day(selected.date)) : left, cursorY = selected ? y(+selected.value) : top;
    const recentDots = observed.length <= 32 ? observed.map(row => `<circle class="lab-plot-observed-dot" cx="${x(day(row.date))}" cy="${y(+row.value)}" r="3"><title>${escape(dateLabel(row.date))}: ${escape(print(row.value))} observed</title></circle>`).join('') : '';
    const validationKey = `<g id="forecast-plot-selection" ${selected ? '' : 'visibility="hidden"'}><line x1="${cursorX}" x2="${cursorX}" y1="${top}" y2="${bottom}"/><circle class="lab-cursor-halo" cx="${cursorX}" cy="${cursorY}" r="10"/><circle cx="${cursorX}" cy="${cursorY}" r="5"/></g>`;
    const latest = observed.at(-1);
    const readout = `<dl class="lab-plot-readout">
      <div><dt><i class="lab-readout-marker lab-readout-observed"></i>Last observation</dt><dd><strong>${print(latest?.value)}</strong><span>${escape(latest ? dateLabel(latest.date) : 'No observation in this view')}</span></dd></div>
      <div><dt><i class="lab-readout-marker lab-readout-estimate"></i>Selected estimate</dt><dd><strong data-lab-selected-value>${print(selected?.value)}</strong><span data-lab-selected-date>${escape(selected ? dateLabel(selected.date) : 'No estimate for this date')}</span></dd></div>
    </dl>`;
    return `<div class="lab-chart-frame">${readout}<div class="lab-plot-shell"><svg id="forecast-plot" viewBox="0 0 884 304" role="img" aria-label="Recorded observations and ${escape(modelName)} forecast over time" data-begin="${begin}" data-end="${end}" data-low="${low}" data-high="${high}">
      ${predicted.length ? `<rect class="lab-plot-future" x="${divider}" y="${top}" width="${extension-divider}" height="${bottom-top}"/>${extended ? `<rect class="lab-plot-extended" x="${extension}" y="${top}" width="${right-extension}" height="${bottom-top}"/>` : ''}` : ''}
      ${ticks}${timeTicks}${bands}${predicted.length ? `<line class="lab-plot-cutoff" x1="${divider}" y1="${top}" x2="${divider}" y2="${bottom}"/>` : ''}
      <path class="lab-plot-observed" d="${path(observed,'value')}"/>${recentDots}<path class="lab-plot-predicted" d="${path(availableAtCutoff ? [observed.at(-1),...predicted] : predicted,'value',1)}"/>${validationKey}</svg></div>
      <div class="forecast-plot-legend"><span><i class="lab-key-observed"></i>Observed</span><span><i class="lab-key-predicted"></i>${escape(modelName)} estimate</span>${intervals.length ? '<span><i class="lab-key-band"></i>Empirical range</span>' : ''}${extended ? '<span><i class="lab-key-extended"></i>Extended projection</span>' : ''}<span class="forecast-plot-caption">Dashed divider: latest dataset date</span></div></div>`;
  }

  function movePlotSelection(svg, point) {
    const selection = svg?.querySelector('#forecast-plot-selection');
    if (!selection) return;
    const frame = svg.closest('.lab-chart-frame');
    const value = frame?.querySelector('[data-lab-selected-value]'), stamp = frame?.querySelector('[data-lab-selected-date]');
    if (value) value.textContent = print(point?.value);
    if (stamp) stamp.textContent = point ? dateLabel(point.date) : 'No estimate for this date';
    selection.setAttribute('visibility', point ? 'visible' : 'hidden');
    if (!point) return;
    const begin = Number(svg.dataset.begin), end = Number(svg.dataset.end), low = Number(svg.dataset.low), high = Number(svg.dataset.high);
    const x = 66 + (day(point.date) - begin) / (end - begin || 1) * 794, y = 255 - (point.value - low) / (high - low) * 223;
    if (!Number.isFinite(x) || !Number.isFinite(y)) { selection.setAttribute('visibility', 'hidden'); return; }
    const line = selection.querySelector('line');
    line.setAttribute('x1', x); line.setAttribute('x2', x);
    selection.querySelectorAll('circle').forEach(marker => { marker.setAttribute('cx', x); marker.setAttribute('cy', y); });
  }

  function renderBenchmark({benchmark, zone, zoneLabel, parameter, parameterLabel, horizon, comparison = 'validation'} = {}) {
    if (!benchmark?.available || !Array.isArray(benchmark.models) || !benchmark.models.length) return '';
    const selectionScore = model => model?.selection_score ?? model?.validation_score;
    const models = [...benchmark.models].sort((first, second) => (numeric(selectionScore(first)) ? +selectionScore(first) : Infinity) - (numeric(selectionScore(second)) ? +selectionScore(second) : Infinity));
    const selected = models.find(model => model.id === benchmark.selected_model), selectedName = benchmark.selected_model_name || selected?.name || 'Selected model';
    const metrics = (benchmark.metrics || []).filter(metric => String(metric.zone_id) === zone && metric.parameter === parameter && +metric.horizon_days === horizon);
    const scoped = model => metrics.find(metric => metric.model === model.id);
    const denominator = Math.max(.001,...models.map(model => numeric(selectionScore(model)) ? +selectionScore(model) : 0));
    const scope = benchmark.selection_scope === 'all' ? 'Whole lake' : 'Pooled zones';
    const scopedBest = models.filter(model => numeric(scoped(model)?.rmse)).sort((first, second) => scoped(first).rmse - scoped(second).rmse)[0];
    const rows = models.map(model => {
      const metric = scoped(model), isSelected = model.id === benchmark.selected_model, score = selectionScore(model), change = metric?.rmse_vs_persistence_pct;
      return `<tr class="${isSelected ? 'model-lab-selected' : ''}"><th scope="row">${escape(model.name || model.id)} ${isSelected ? '<span class="model-lab-label">Auto</span>' : ''}</th><td><div class="model-lab-score">${print(score, 4)}${numeric(score) ? `<span class="model-lab-meter" aria-hidden="true"><i style="width:${Math.max(0,Math.min(100,+score / denominator * 100))}%"></i></span>` : ''}</div></td><td>${print(metric?.rmse, 4)}</td><td class="${numeric(metric?.r2) && +metric.r2 < 0 ? 'model-lab-negative' : ''}">${print(metric?.r2, 3)}</td><td class="${numeric(change) && +change < 0 ? 'model-lab-negative' : ''}">${numeric(change) ? `${+change > 0 ? '+' : ''}${print(change,1)}%` : '—'}</td><td>${print(metric?.n, 0)}</td></tr>`;
    }).join('');
    const caveats = Array.isArray(benchmark.limitations) ? benchmark.limitations : benchmark.limitations ? [benchmark.limitations] : [];
    return `<section class="card model-lab" aria-labelledby="model-lab-title"><div class="model-lab-header"><div><div class="eyebrow">MODEL COMPARISON</div><h2 id="model-lab-title" class="card-title">Compare the forecasting methods</h2><p>Auto uses earlier ${scope.toLowerCase()} validation scores. Select a location, indicator and horizon to see how the methods compare on later dates.</p></div><div class="model-lab-winner"><span>AUTO SELECTED</span><strong>${escape(selectedName)}</strong>${numeric(selectionScore(selected)) ? `<small>${print(selectionScore(selected),4)} normalized RMSE</small>` : ''}</div></div>
      <div class="model-lab-scope"><span>${escape(zoneLabel)}</span><span>${escape(parameterLabel)}</span><span>≈ ${escape(horizon)} days</span><span>Full retrospective period</span></div>
      <div class="table-wrap model-lab-table" role="region" tabindex="0" aria-label="Candidate benchmarks; scroll horizontally for all metrics"><table class="data-table"><thead><tr><th scope="col">Candidate</th><th scope="col" title="Aggregated validation score across indicators and horizons; lower is better">Validation score</th><th scope="col" title="Root mean squared error on the retrospective dates; lower is better">Test RMSE</th><th scope="col">Test R²</th><th scope="col" title="Positive values mean the model reduced RMSE compared with carrying forward the latest reading">vs. last reading</th><th scope="col">Pairs</th></tr></thead><tbody>${rows}</tbody></table></div>
      ${scopedBest ? `<p class="model-lab-result"><strong>Lowest test error here:</strong> ${escape(scopedBest.name || scopedBest.id)} (${print(scoped(scopedBest).rmse,4)} RMSE).</p>` : ''}
      <p class="model-lab-foot">Validation score balances different indicator scales; lower is better. Positive percentages mean lower test RMSE than simply using the last reading. ${metrics.length ? '' : 'This scope has no comparable test metrics. '}${comparison === 'validation' ? 'Changing the chart date range does not change the full-period comparison above.' : 'This comparison uses the closest tested horizon; longer projections are experimental.'}</p>
      <details class="provenance"><summary>Selection method and limits</summary>${benchmark.method ? `<p>${escape(benchmark.method)}</p>` : ''}<p>Training through ${escape(dateLabel(benchmark.train_end))}; selection through ${escape(dateLabel(benchmark.validation_end))}. Retrospective test: ${escape(dateLabel(benchmark.test_start))} to ${escape(dateLabel(benchmark.test_end))}.</p>${caveats.map(caveat => `<p>${escape(caveat)}</p>`).join('')}</details></section>`;
  }
  function render(context = {}) {
    const c = context, meta = c.meta || {}, date = c.helpers?.date || dateLabel, icon = c.helpers?.ico || (() => '');
    const horizon = Number.isFinite(Number(c.horizon)) ? Number(c.horizon) : 30;
    const trained = Boolean(c.modelAvailable), validated = Number.isFinite(+c.validatedDays) && +c.validatedDays > 0 ? +c.validatedDays : 32, extended = horizon > validated;
    const parameter = c.parameterLabel || 'Selected indicator', zone = c.zoneLabel || 'Whole lake';
    const model = c.modelName || 'DET';
    const quickDates = [...new Set([8,16,32].filter(days => days <= horizon).concat(horizon))].map(days => `<button class="btn btn-small" type="button" data-forecast-offset="${days}" aria-label="Show forecast ${days} days after the latest observation">${days} days</button>`).join('');
    return `
      <section class="outlook-hero" aria-labelledby="outlook-hero-title">
        <div class="outlook-hero-copy">
          <div class="outlook-kicker">FORECAST STUDIO</div>
          <h2 id="outlook-hero-title">From readings to outlook.</h2>
          <p>See how the selected model expects this indicator to change. Pick a date for a closer look.</p>
          <div class="lab-hero-scope"><span>${escape(zone)}</span><span>${escape(parameter)}</span><span class="badge ${extended ? 'warning' : 'neutral'}">${extended ? `Beyond ${validated} days · experimental projection` : 'Short range research forecast'}</span></div>
        </div>
        <dl class="outlook-hero-facts" aria-label="Forecast window and method">
          <div><dt>Last observed</dt><dd><strong>${escape(date(c.latestDate))}</strong></dd></div>
          <div><dt>Outlook through</dt><dd><strong>${escape(date(c.forecastEnd))}</strong></dd></div>
          <div><dt>Forecast method</dt><dd><strong id="forecast-model-fact">${escape(model)}</strong><small>${trained ? 'Ready for this dataset' : 'Train or choose a compatible method'}</small></dd></div>
        </dl>
      </section>

      <section class="card forecast-workspace" aria-labelledby="forecast-chart-title">
        <div class="card-header forecast-section-heading">
          <div><span class="eyebrow">FORECAST TIMELINE</span><h2 class="card-title" id="forecast-chart-title">${escape(parameter)} over time</h2><p class="card-subtitle">Historical readings and future estimates · ${escape(zone)}</p></div>
        </div>
        <div class="forecast-chart-controls"><label class="forecast-location-label"><span>Location</span><select class="select" id="zone-select" aria-label="Select zone">${c.zoneOptionsMarkup || '<option value="all">All zones</option>'}</select></label><label><span>Forecast model</span><select class="select" id="forecast-model" aria-label="Forecast model">${c.modelOptionsMarkup || '<option value="det">DET</option>'}</select></label><label><span>Observed history</span><select class="select" id="forecast-history" aria-label="Observed history shown">${[[12,'Last 12 readings'],[24,'Last 24 readings'],[48,'Last 48 readings'],[0,'All readings']].map(([count,label]) => `<option value="${count}" ${count === (c.historyCount ?? 24) ? 'selected' : ''}>${label}</option>`).join('')}</select></label><label class="forecast-band-control"><input type="checkbox" id="forecast-band" ${c.showBand === false ? '' : 'checked'}><span>Show empirical range</span></label></div>
        ${c.qualityMarkup || ''}
        <div id="forecast-chart"><div id="forecast-plot-body"><div class="loading" role="status">Preparing the forecast chart…</div></div><div id="forecast-zone-detail"></div></div>
        <div class="forecast-scrubber"><label for="forecast-scrub"><strong>Inspect forecast day</strong><output id="forecast-scrub-label" for="forecast-scrub">${escape(date(c.selectedDate))}</output></label><input type="range" id="forecast-scrub" min="1" max="${horizon}" value="${Math.max(1,Math.min(horizon, Math.round((day(c.selectedDate)-day(c.latestDate))/86400000) || 1))}" aria-valuetext="${escape(date(c.selectedDate))}" ${trained ? '' : 'disabled'}><div class="forecast-scrub-labels"><span>First forecast day</span><span>Day ${horizon}</span></div></div>
        <div class="forecast-footer"><span class="chip">${icon('clock')} ${horizon}-day outlook</span><span class="muted" id="interval-status">Forecast starts after ${escape(date(c.latestDate))}</span><button class="btn" id="download-forecast" disabled>${icon('download')} Download forecast CSV</button></div>
      </section>

      <section class="card forecast-date-card" aria-labelledby="forecast-date-title">
        <div class="card-header forecast-section-heading">
          <div><span class="eyebrow">DATE INSPECTOR</span><h2 class="card-title" id="forecast-date-title">One date. All three indicators.</h2><p class="card-subtitle">${escape(zone)} · Estimated values in your uploaded indicator scale</p></div>
          <div class="forecast-date-controls"><label><span>Prediction date</span><input class="select" id="prediction-date" type="date" min="${escape(c.minDate)}" max="${escape(c.forecastEnd)}" value="${escape(c.selectedDate)}" aria-label="Select future prediction date"></label><div class="forecast-quick-dates" aria-label="Quick prediction dates">${quickDates}</div></div>
        </div>
        <div id="prediction-cards" class="prediction-grid outlook-summary-grid"><div class="empty-state">Loading estimates for all three indicators…</div></div>
        <p class="fine-print">The change shown on each card is relative to the latest observed value. These are index/proxy estimates, not confirmed water-use or health classifications.</p>
        <details class="forecast-method provenance"><summary>How to read daily estimates and uncertainty</summary><p>Accuracy is compared near 8, 16 and 32 days when outcomes are available. Daily estimates between these horizons do not have separate validation. Projections beyond ${validated} days are experimental.</p><p>Shading is displayed only where the selected method provides an empirical range. It reflects previously observed forecast errors; it does not guarantee future coverage.</p></details>
      </section>

      ${c.comparisonMarkup || ''}
      <div class="forecast-insight-grid section-grid">
        <section class="card" aria-labelledby="forecast-accuracy-title">
          <div class="card-header"><div><span class="eyebrow">ERROR PROFILE</span><h2 class="card-title" id="forecast-accuracy-title">How well has it predicted?</h2><p class="card-subtitle">${escape(parameter)} · ${escape(zone)} · ${escape(model)}</p></div><button class="btn btn-small" type="button" id="forecast-open-accuracy">Review actual readings</button></div>
          ${c.benchmarkMarkup || '<div class="empty-state">There are no matched historical scores for this model and dataset.</div>'}
          <dl class="forecast-metric-guide"><div><dt>MAE / RMSE</dt><dd>Prediction error. Lower values are better.</dd></div><div><dt>R²</dt><dd>Agreement against an average-value reference. Negative values mean the model performed worse than that reference.</dd></div></dl>
          <p class="fine-print">The test readings were excluded from fitting and model selection. This historical period has already been reviewed during development. These results do not establish operational accuracy.</p>
        </section>
        <section class="card forecast-model-card" aria-labelledby="forecast-model-title">
          <div class="card-header"><div><span class="eyebrow">MODEL WORKSPACE</span><h2 class="card-title" id="forecast-model-title">DET training</h2><p class="card-subtitle">${c.detAvailable ?? trained ? 'A compatible trained DET is available.' : escape(c.detReason || 'Train DET to make it available for this dataset.')}</p></div>${icon('layers')}</div>
          <p class="muted">A new or changed dataset needs its own compatible model. Training runs in the background and shows its progress.</p>
          <button class="btn btn-primary" id="train-model">${icon('play')} ${c.detAvailable ?? trained ? 'Retrain DET' : 'Train DET on this dataset'}</button><div id="training-live"></div>
          <details class="forecast-training-details provenance"><summary>What DET learns</summary><div class="hierarchy-diagram"><span>Each location's ordered readings</span><i>↓</i><div><b>Cells</b><b>Zones</b><b>Lake</b></div><i>↓</i><span>Estimated future indicator values</span></div><p>DET learns individual cell histories and combines them through a geographic tree. The model must match the uploaded observations.</p></details>
        </section>
      </div>

      <section class="card forecast-evidence-card" aria-labelledby="forecast-evidence-title">
        <div class="card-header"><div><span class="eyebrow">SUPPORTING EVIDENCE</span><h2 class="card-title" id="forecast-evidence-title">Evidence behind the outlook</h2><p class="card-subtitle">Observed conditions, model estimates, and references you can inspect</p></div><button class="btn" id="download-explanation" disabled>${icon('download')} Download explanation</button></div>
        <p class="forecast-reading-guide">The forecast model produces the numbers. Retrieved references help interpret the observations and do not prove a local cause.</p>
        <div id="explanation-content"><div class="loading" role="status">Retrieving the relevant reference passages…</div></div>
      </section>

      <details class="card forecast-method" aria-labelledby="forecast-record-title">
        <summary id="forecast-record-title">Training and evaluation details</summary>
        <div class="card-header"><div><h2 class="card-title">The chronological model record</h2><p class="card-subtitle">Earlier dates fit the model; validation dates select it; later dates evaluate it.</p></div><span class="badge neutral">${escape(meta.source || c.sourceLabel || 'Uploaded observations')}</span></div>
        <div class="stat-grid"><div><span class="muted">Train through</span><strong>${escape(date(meta.train_end))}</strong></div><div><span class="muted">Validate through</span><strong>${escape(date(meta.validation_end))}</strong></div><div><span class="muted">Test from</span><strong>${escape(date(meta.test_start))}</strong></div><div><span class="muted">Test windows</span><strong>${escape(meta.test_windows ?? '—')}</strong></div></div>
        <p class="fine-print">${escape(meta.evaluation || 'Train DET on sufficient historical observations to create an evaluation record.')}</p>
      </details>`;
  }
  window.AquaForecastUI = Object.freeze({render, plot, movePlotSelection, renderBenchmark});
})();
