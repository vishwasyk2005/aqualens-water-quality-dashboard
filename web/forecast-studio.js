/* Presentation-only forecast studio. Scientific values are supplied by app.js. */
(() => {
  'use strict';
  const escape = value => String(value ?? '').replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
  const dateLabel = value => {
    const date = new Date(String(value ?? '').slice(0, 10) + 'T12:00:00');
    return Number.isFinite(+date) ? date.toLocaleDateString('en-GB', {day:'numeric', month:'short', year:'numeric'}) : 'Not available';
  };
  function render(context = {}) {
    const c = context, meta = c.meta || {}, date = c.helpers?.date || dateLabel, icon = c.helpers?.ico || (() => '');
    const horizon = Number.isFinite(Number(c.horizon)) ? Number(c.horizon) : 30;
    const trained = Boolean(c.modelAvailable), extended = horizon > 32;
    const parameter = c.parameterLabel || 'Selected indicator', zone = c.zoneLabel || 'Whole lake';
    const quickDates = [8, 16, 30].filter(days => days <= horizon).map(days => `<button class="btn btn-small" type="button" data-forecast-offset="${days}" aria-label="Show forecast ${days} days after the latest observation">${days} days</button>`).join('');
    return `
      <section class="outlook-hero" aria-labelledby="outlook-hero-title">
        <div class="outlook-hero-copy">
          <div class="outlook-kicker">FROM YOUR OBSERVATIONS TO A FUTURE OUTLOOK</div>
          <h2 id="outlook-hero-title">What could change next?</h2>
          <div class="outlook-current">${escape(zone)} <span>·</span> ${escape(parameter)}</div>
          <p>Explore estimates from the trained DET model. Start with a location, choose an indicator, then inspect the date that matters to you.</p>
          <span class="badge ${extended ? 'warning' : 'neutral'}">${extended ? 'Beyond 32 days · experimental projection' : 'Short-horizon research forecast'}</span>
        </div>
        <div class="outlook-hero-facts">
          <div><span>LAST OBSERVED</span><strong>${escape(date(c.latestDate))}</strong><small>Latest uploaded measurement</small></div>
          <div><span>OUTLOOK THROUGH</span><strong>${escape(date(c.forecastEnd))}</strong><small>${horizon} days after the last observation</small></div>
          <div><span>MODEL</span><strong>DET</strong><small>${trained ? 'Trained on this dataset' : 'Training required for this dataset'}</small></div>
        </div>
      </section>

      <section class="card forecast-workspace" aria-labelledby="forecast-chart-title">
        <div class="card-header forecast-section-heading">
          <div><span class="eyebrow">01 / EXPLORE THE OUTLOOK</span><h2 class="card-title" id="forecast-chart-title">${escape(parameter)} over time</h2><p class="card-subtitle">Historical readings and future estimates · ${escape(zone)}</p></div>
          <div class="toolbar"><label class="forecast-location-label"><span>Location</span><select class="select" id="zone-select" aria-label="Select zone">${c.zoneOptionsMarkup || '<option value="all">All zones</option>'}</select></label><span class="badge ${trained ? 'normal' : 'neutral'}">${icon('layers')} ${trained ? 'DET ready' : 'Train DET first'}</span></div>
        </div>
        ${c.qualityMarkup || ''}
        <p class="forecast-reading-guide">The solid line shows recorded observations. The dashed line shows DET estimates after the last observation date.</p>
        <div id="forecast-chart"><div class="loading" role="status">Preparing the forecast chart…</div></div>
        <div class="forecast-footer"><span class="chip">${icon('clock')} ${horizon}-day outlook</span><span class="muted" id="interval-status">Forecast starts after ${escape(date(c.latestDate))}</span><button class="btn" id="download-forecast" disabled>${icon('download')} Download forecast CSV</button></div>
      </section>

      <section class="card forecast-date-card" aria-labelledby="forecast-date-title">
        <div class="card-header forecast-section-heading">
          <div><span class="eyebrow">02 / CHECK A PARTICULAR DATE</span><h2 class="card-title" id="forecast-date-title">One date. All three indicators.</h2><p class="card-subtitle">${escape(zone)} · Estimated values in your uploaded indicator scale</p></div>
          <div class="forecast-date-controls"><label><span>Prediction date</span><input class="select" id="prediction-date" type="date" min="${escape(c.minDate)}" max="${escape(c.forecastEnd)}" value="${escape(c.selectedDate)}" aria-label="Select future prediction date"></label><div class="forecast-quick-dates" aria-label="Quick prediction dates">${quickDates}</div></div>
        </div>
        <div id="prediction-cards" class="prediction-grid outlook-summary-grid"><div class="empty-state">Loading estimates for all three indicators…</div></div>
        <p class="fine-print">The change shown on each card is relative to the latest observed value. These are index/proxy estimates, not confirmed water-use or health classifications.</p>
        <details class="forecast-method provenance"><summary>How to read daily estimates and uncertainty</summary><p>DET has native forecast steps of approximately 8, 16 and 32 days. Daily values between those steps are interpolated. Projections beyond 32 days are experimental and do not inherit short-horizon accuracy.</p><p>Where shown, empirical zone bands come from the evaluation model's validation errors and are transferred to the model refitted on all history. Test coverage can be below 80%; future coverage is not guaranteed. A whole-lake interval is not calibrated.</p></details>
      </section>

      <div class="forecast-insight-grid section-grid">
        <section class="card" aria-labelledby="forecast-accuracy-title">
          <div class="card-header"><div><span class="eyebrow">03 / UNDERSTAND THE ERRORS</span><h2 class="card-title" id="forecast-accuracy-title">How well has it predicted?</h2><p class="card-subtitle">${escape(parameter)} · ${escape(zone)} · Retrospective test</p></div><button class="btn btn-small" type="button" data-open-page="validation">Open accuracy checks ${icon('arrow')}</button></div>
          ${c.benchmarkMarkup || '<div class="empty-state">Train DET to create an evaluation record for this dataset.</div>'}
          <dl class="forecast-metric-guide"><div><dt>MAE / RMSE</dt><dd>Prediction error. Lower values are better.</dd></div><div><dt>R²</dt><dd>Agreement against an average-value reference. Negative values mean the model performed worse than that reference.</dd></div></dl>
          <p class="fine-print">The test readings were excluded from fitting and model selection. This historical period has already been reviewed during development. These results do not establish operational accuracy.</p>
        </section>
        <section class="card forecast-model-card" aria-labelledby="forecast-model-title">
          <div class="card-header"><div><span class="eyebrow">MODEL WORKSPACE</span><h2 class="card-title" id="forecast-model-title">A model for your observations</h2><p class="card-subtitle">${trained ? 'A compatible trained DET is available.' : escape(c.modelReason || 'Train DET before predicting this dataset.')}</p></div>${icon('layers')}</div>
          <p class="muted">A new or changed dataset needs its own compatible model. Training runs in the background and shows its progress.</p>
          <button class="btn btn-primary" id="train-model">${icon('play')} ${trained ? 'Retrain DET' : 'Train DET on this dataset'}</button><div id="training-live"></div>
          <details class="forecast-training-details provenance"><summary>What DET learns</summary><div class="hierarchy-diagram"><span>Each location's ordered readings</span><i>↓</i><div><b>Cells</b><b>Zones</b><b>Lake</b></div><i>↓</i><span>Estimated future indicator values</span></div><p>DET learns individual cell histories and combines them through a geographic tree. The model must match the uploaded observations.</p></details>
        </section>
      </div>

      <section class="card forecast-evidence-card" aria-labelledby="forecast-evidence-title">
        <div class="card-header"><div><span class="eyebrow">04 / READ THE CONTEXT</span><h2 class="card-title" id="forecast-evidence-title">Evidence behind the outlook</h2><p class="card-subtitle">Observed conditions, model estimates, and references you can inspect</p></div><button class="btn" id="download-explanation">${icon('download')} Download explanation</button></div>
        <p class="forecast-reading-guide">DET produces the numerical prediction. Retrieved references support its interpretation and do not prove a local cause.</p>
        <div id="explanation-content"><div class="loading" role="status">Retrieving the relevant reference passages…</div></div>
      </section>

      <details class="card forecast-method" aria-labelledby="forecast-record-title">
        <summary id="forecast-record-title">Training and evaluation details</summary>
        <div class="card-header"><div><h2 class="card-title">The chronological model record</h2><p class="card-subtitle">Earlier dates fit the model; validation dates select it; later dates evaluate it.</p></div><span class="badge neutral">${escape(meta.source || c.sourceLabel || 'Uploaded observations')}</span></div>
        <div class="stat-grid"><div><span class="muted">Train through</span><strong>${escape(date(meta.train_end))}</strong></div><div><span class="muted">Validate through</span><strong>${escape(date(meta.validation_end))}</strong></div><div><span class="muted">Test from</span><strong>${escape(date(meta.test_start))}</strong></div><div><span class="muted">Test windows</span><strong>${escape(meta.test_windows ?? '—')}</strong></div></div>
        <p class="fine-print">${escape(meta.evaluation || 'Train DET on sufficient historical observations to create an evaluation record.')}</p>
      </details>`;
  }
  window.AquaForecastUI = Object.freeze({render});
})();
