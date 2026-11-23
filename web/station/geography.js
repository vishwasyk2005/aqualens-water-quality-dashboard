import {escapeHtml, formatDate, formatNumber, icon, parameterNames} from './utils.js';

const PARAMETERS = ['ndwi', 'turbidity_proxy', 'chlorophyll_proxy'];
const LABELS = {...parameterNames, ndwi: 'NDWI'};
const STOPS = ['#127b80', '#78c78e', '#e7ee88'];
const LAYERS = {streets: 'Streets', satellite: 'Satellite', plain: 'Plain'};
const indexedRows = new WeakMap();
const day = value => String(value ?? '').slice(0, 10);
const numeric = value => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));
const coordinate = row => numeric(row.lat) && numeric(row.lon) && Math.abs(+row.lat) <= 90 && Math.abs(+row.lon) <= 180;
const displayName = row => String(row.zone_name || row.zone_id || 'Unassigned');
const count = value => formatNumber(value, 0);
const observed = (row, parameter) => numeric(row[parameter])
  && (parameter !== 'ndwi' || Math.abs(+row[parameter]) <= 1)
  && !['false', '0', 'no'].includes(String(row[parameter + '_is_observed'] ?? row.is_observed).toLowerCase())
  && (row.valid_pixel_fraction == null || +row.valid_pixel_fraction >= .5);
const rowKey = row => JSON.stringify([row.cell_id, row.lat, row.lon, row.zone_id]);

function index(rows) {
  const cached = indexedRows.get(rows);
  if (cached) return cached;
  const byDate = new Map(), zones = new Map(), bounds = [Infinity, Infinity, -Infinity, -Infinity];
  for (const row of rows) {
    const date = day(row.date);
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !coordinate(row)) continue;
    if (!byDate.has(date)) byDate.set(date, []);
    byDate.get(date).push(row);
    if (row.zone_id != null) zones.set(String(row.zone_id), displayName(row));
    bounds[0] = Math.min(bounds[0], +row.lat);
    bounds[1] = Math.min(bounds[1], +row.lon);
    bounds[2] = Math.max(bounds[2], +row.lat);
    bounds[3] = Math.max(bounds[3], +row.lon);
  }
  const value = {byDate, zones, bounds, dates: [...byDate.keys()].sort()};
  indexedRows.set(rows, value);
  return value;
}

function extent(rows, parameter) {
  let min = Infinity, max = -Infinity;
  for (const row of rows) {
    min = Math.min(min, +row[parameter]);
    max = Math.max(max, +row[parameter]);
  }
  return [min, max];
}

function color(value, min, max) {
  const fraction = max > min ? Math.max(0, Math.min(1, (value - min) / (max - min))) : .5;
  const low = fraction < .5 ? STOPS[0] : STOPS[1], high = fraction < .5 ? STOPS[1] : STOPS[2];
  const mix = fraction < .5 ? fraction * 2 : (fraction - .5) * 2;
  const channel = offset => Math.round(parseInt(low.slice(offset, offset + 2), 16) * (1 - mix) + parseInt(high.slice(offset, offset + 2), 16) * mix);
  return `rgb(${channel(1)}, ${channel(3)}, ${channel(5)})`;
}

function options(items, selected) {
  return items.map(([id, label]) => `<option value="${escapeHtml(id)}"${id === selected ? ' selected' : ''}>${escapeHtml(label)}</option>`).join('');
}

export function renderGeography(host, ctx) {
  const rows = Array.isArray(ctx.data?.observations) ? ctx.data.observations : [];
  const {byDate, dates, zones, bounds} = index(rows);
  const dataset = String(ctx.data?.dataset ?? '');
  if (ctx.state.geography?.dataset !== dataset) ctx.state.geography = {dataset, parameter: 'ndwi', layer: 'streets', zone: 'all'};
  const state = ctx.state.geography;
  if (ctx.state.targetZone && (zones.has(String(ctx.state.targetZone)) || ctx.state.targetZone === 'all')) {
    state.zone = String(ctx.state.targetZone);
    delete ctx.state.targetZone;
  }
  if (!zones.has(state.zone) && state.zone !== 'all') state.zone = 'all';
  if (!PARAMETERS.includes(state.parameter)) state.parameter = 'ndwi';
  if (!Object.hasOwn(LAYERS, state.layer)) state.layer = 'streets';
  if (!dates.includes(state.date)) state.date = dates.at(-1) || '';
  const title = ctx.route === 'zones' ? 'Explore a monitoring zone' : 'Lake observation map';
  host.innerHTML = `<section class="terrain-workspace" aria-labelledby="terrain-title">
    <header class="terrain-heading">
      <div class="terrain-head-copy"><p class="terrain-kicker">${icon('map', 16)} Spatial explorer</p>
      <h1 id="terrain-title">${title}</h1>
      <p>Explore where each satellite reading was taken. Change the date or indicator, then select a point to see the values reported at that location.</p></div>
      <div class="terrain-head-stats" aria-label="Dataset coverage"><div><b>${count(zones.size)}</b><span>Zones</span></div><div><b>${count(dates.length)}</b><span>Mapped dates</span></div></div>
    </header>
    ${dates.length ? `<div class="terrain-frame">
      <header class="terrain-toolbar">
        <div class="terrain-control-row">
          <label><span>Indicator</span><select data-terrain-parameter>${options(PARAMETERS.map(key => [key, LABELS[key]]), state.parameter)}</select></label>
          <label><span>Monitor area</span><select data-terrain-zone>${options([['all', 'Whole lake'], ...zones], state.zone)}</select></label>
          <div class="terrain-layer-field"><span id="terrain-basemap-label">Map background</span><div class="terrain-layer-control" role="group" aria-labelledby="terrain-basemap-label">
            ${Object.entries(LAYERS).map(([key, name]) => `<button type="button" data-terrain-layer="${key}" aria-pressed="${state.layer === key}">${name}</button>`).join('')}
          </div></div>
        </div>
      </header>
      <div class="terrain-time">
        <div class="terrain-date"><label for="terrain-date-select">${icon('calendar', 17)} Observation date</label><select id="terrain-date-select" data-terrain-date>${options(dates.map(date => [date, formatDate(date)]), state.date)}</select></div>
        <div class="terrain-scrubber">
          <div class="terrain-track"><button type="button" data-terrain-prev aria-label="Previous observation date" title="Previous observation date">‹</button><input type="range" min="0" max="${dates.length - 1}" step="1" value="${dates.indexOf(state.date)}" data-terrain-time aria-label="Browse observation dates"><button type="button" data-terrain-next aria-label="Next observation date" title="Next observation date">›</button></div>
          <div class="terrain-track-dates"><span>${escapeHtml(formatDate(dates[0]))}</span><span>${escapeHtml(formatDate(dates.at(-1)))}</span></div>
        </div>
      </div>
      <div class="terrain-layout">
        <div class="terrain-stage">
          <div class="terrain-canvas" data-terrain-canvas role="region" aria-label="Water observation map. Use arrow keys to pan and plus or minus to zoom."></div>
          <div class="terrain-map-top">
            <span class="terrain-map-pill"><i></i><span data-terrain-count></span></span>
            <div class="terrain-tools"><button type="button" data-terrain-fit title="Fit to readings" aria-label="Fit map to readings">${icon('target', 18)}</button><button type="button" data-terrain-full title="Expand map" aria-label="Expand map" aria-pressed="false">${icon('expand', 18)}</button></div>
          </div>
          <p class="terrain-tile-note" data-terrain-tile-note role="status" hidden>Background tiles are unavailable. You can still explore recorded points using Plain.</p>
          <div class="terrain-map-empty" data-terrain-no-values hidden>No valid readings match these filters.</div>
          <div class="terrain-legend" data-terrain-legend></div>
        </div>
        <aside class="terrain-inspector" aria-label="Reading inspector">
          <div class="terrain-inspector-lead"><span>${icon('pin', 19)}</span><h2>Reading inspector</h2></div>
          <section data-terrain-card></section>
          <details class="terrain-reading-list"><summary>Find a reading</summary>
            <label class="terrain-search">${icon('search', 16)}<input data-terrain-search aria-label="Find a reading by cell ID or zone" placeholder="Cell ID or zone name"></label>
            <div data-terrain-results></div>
          </details>
        </aside>
      </div>
    </div>
    <p class="terrain-note">${icon('info', 15)} <span>Point colours show low to high readings for this view. They do not indicate health or safety. Points require a valid recorded value; absent pixel quality is shown as unknown.</span></p>` :
    `<div class="terrain-empty">${icon('map', 32)}<h2>No georeferenced observations</h2><p>Load a dataset with recorded dates, latitude and longitude to use the map.</p><button type="button" data-terrain-open-data>Open data library</button></div>`}
  </section>`;
  const query = selector => host.querySelector(selector);
  if (!dates.length) {
    query('[data-terrain-open-data]').onclick = () => ctx.navigate('data');
    return;
  }
  const leaflet = window.L;
  const canvas = query('[data-terrain-canvas]');
  const frame = query('.terrain-frame');
  const countLabel = query('[data-terrain-count]');
  const card = query('[data-terrain-card]');
  const results = query('[data-terrain-results]');
  const legend = query('[data-terrain-legend]');
  const empty = query('[data-terrain-no-values]');
  const tileNote = query('[data-terrain-tile-note]');
  const search = query('[data-terrain-search]');
  const expand = query('[data-terrain-full]');
  const dateSelect = query('[data-terrain-date]');
  const range = query('[data-terrain-time]');
  let map, dots, activeTile, ring, resizeObserver, visible = [], selected = -1, redrawFrame = 0;
  let disposed = false;
  const tiles = {};
  if (leaflet) {
    map = leaflet.map(canvas, {zoomControl: false, scrollWheelZoom: false, zoomSnap: .5, minZoom: 1, maxZoom: 19, preferCanvas: true});
    leaflet.control.zoom({position: 'bottomright'}).addTo(map);
    leaflet.control.scale({position: 'bottomright', imperial: false}).addTo(map);
    dots = leaflet.layerGroup().addTo(map);
    tiles.streets = leaflet.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 19, attribution: '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a>', referrerPolicy: 'strict-origin-when-cross-origin'});
    tiles.satellite = leaflet.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {maxZoom: 19, attribution: 'Imagery © Esri, Vantor, Earthstar Geographics, and the GIS User Community', referrerPolicy: 'strict-origin-when-cross-origin'});
    for (const tile of Object.values(tiles)) tile.on('tileerror', () => { if (!disposed && activeTile === tile) tileNote.hidden = false; });
    map.fitBounds([[bounds[0], bounds[1]], [bounds[2], bounds[3]]], {padding: [26, 25], maxZoom: 15, animate: false});
    if (typeof ResizeObserver === 'function') {
      resizeObserver = new ResizeObserver(() => { if (!disposed) map.invalidateSize({pan: false}); });
      resizeObserver.observe(canvas);
    }
  } else {
    query('[data-terrain-fit]').disabled = true;
    canvas.textContent = 'Map graphics are unavailable. You can still find and inspect named readings.';
  }

  function layer(key) {
    state.layer = key;
    tileNote.hidden = true;
    frame.dataset.basemap = key;
    if (activeTile) map.removeLayer(activeTile);
    activeTile = tiles[key];
    if (activeTile) activeTile.addTo(map);
    for (const button of host.querySelectorAll('[data-terrain-layer]')) button.setAttribute('aria-pressed', String(button.dataset.terrainLayer === key));
  }

  function fit() {
    if (!map) return;
    const candidates = visible.length ? visible : (byDate.get(state.date) || []).filter(row => state.zone === 'all' || String(row.zone_id) === state.zone);
    if (candidates.length > 1) {
      const viewBounds = leaflet.latLngBounds([]);
      for (const row of candidates) viewBounds.extend([row.lat, row.lon]);
      map.fitBounds(viewBounds, {paddingTopLeft: [24, 34], paddingBottomRight: [24, 82], maxZoom: 15, animate: false});
    } else if (candidates.length === 1) {
      map.setView([candidates[0].lat, candidates[0].lon], 14);
    } else {
      map.fitBounds([[bounds[0], bounds[1]], [bounds[2], bounds[3]]], {padding: [26, 25], maxZoom: 15, animate: false});
    }
  }

  function detail(position, pan) {
    const row = visible[position];
    if (!row) return;
    selected = position;
    state.cell = rowKey(row);
    if (map && ring) { map.removeLayer(ring); ring = null; }
    if (map) {
      ring = leaflet.circleMarker([row.lat, row.lon], {radius: 8, stroke: true, color: '#0c2924', weight: 2.5, fill: false, interactive: false}).addTo(map);
      if (pan && !map.getBounds().pad(-.1).contains([+row.lat, +row.lon])) map.panTo([+row.lat, +row.lon], {animate: false});
    }
    const qualityKnown = numeric(row.valid_pixel_fraction);
    card.innerHTML = `<div class="terrain-point-meta">${escapeHtml(formatDate(row.date))} <span>Recorded</span></div>
      <h3>${escapeHtml(displayName(row))}</h3>
      <p class="terrain-cell" title="${escapeHtml(row.cell_id)}">${escapeHtml(row.cell_id || 'Cell ID unavailable')}</p>
      <dl class="terrain-values">${PARAMETERS.map(parameter => `<div${parameter === state.parameter ? ' class="terrain-highlight"' : ''}><dt>${escapeHtml(LABELS[parameter])}</dt><dd>${observed(row, parameter) ? formatNumber(row[parameter], 3) : '—'}</dd></div>`).join('')}</dl>
      <p class="terrain-reading-note">${icon('globe', 14)} <span>${formatNumber(row.lat, 5)}°, ${formatNumber(row.lon, 5)}°</span></p>
      <p class="terrain-reading-note">${icon('info', 14)} <span>${qualityKnown ? `Valid pixels: ${formatNumber(+row.valid_pixel_fraction * 100, 1)}%` : 'Pixel quality was not supplied.'}</span></p>`;
    updateSearch();
  }

  function updateSearch() {
    const term = search.value.trim().toLowerCase();
    const found = [];
    let total = 0;
    for (let position = 0; position < visible.length; position++) {
      const row = visible[position];
      if (term && ![row.cell_id, row.zone_id, row.zone_name].some(value => String(value ?? '').toLowerCase().includes(term))) continue;
      total++;
      if (found.length < 10) found.push([position, row]);
    }
    results.innerHTML = found.length ? `${found.map(([position,row]) => `<button class="terrain-result" data-terrain-reading="${position}" type="button" aria-pressed="${position === selected}"><span><strong>${escapeHtml(displayName(row))}</strong><small>${escapeHtml(row.cell_id || 'Point')}</small></span><b>${formatNumber(row[state.parameter], 3)}</b></button>`).join('')}
      ${total > found.length ? `<p class="terrain-list-note">${count(total)} matching points. Search to narrow the list.</p>` : ''}` : '<p class="terrain-list-note">No matching points.</p>';
  }

  function redraw() {
    if (disposed) return;
    const parameter = state.parameter;
    const position = dates.indexOf(state.date);
    dateSelect.value = state.date;
    range.value = String(position);
    range.setAttribute('aria-valuetext', formatDate(state.date));
    query('[data-terrain-prev]').disabled = position <= 0;
    query('[data-terrain-next]').disabled = position >= dates.length - 1;
    visible = (byDate.get(state.date) || []).filter(row => (state.zone === 'all' || String(row.zone_id) === state.zone) && observed(row, parameter));
    selected = -1;
    if (map && ring) { map.removeLayer(ring); ring = null; }
    if (dots) dots.clearLayers();
    const [min, max] = extent(visible, parameter);
    for (let choice = 0; dots && choice < visible.length; choice++) {
      const row = visible[choice];
      const marker = leaflet.circleMarker([+row.lat, +row.lon], {
        radius: visible.length > 1000 ? 3.4 : 4.4, weight: .85, color: '#113530',
        fillColor: color(+row[parameter], min, max), fillOpacity: .94
      });
      marker.bindTooltip(`${escapeHtml(displayName(row))} · ${formatNumber(row[parameter], 3)}`, {direction: 'top', opacity: 1});
      marker.on('click', () => detail(choice, false));
      marker.addTo(dots);
    }
    countLabel.textContent = `${count(visible.length)} ${visible.length === 1 ? 'reading' : 'readings'}`;
    empty.hidden = visible.length > 0;
    legend.hidden = !visible.length;
    legend.innerHTML = visible.length ? `<strong>${escapeHtml(LABELS[parameter])}</strong><div class="terrain-ramp"></div><div class="terrain-range-labels"><span>${formatNumber(min, 3)}</span><span>${formatNumber(max, 3)}</span></div>` : '';
    const retained = state.cell == null ? -1 : visible.findIndex(row => rowKey(row) === state.cell);
    if (retained >= 0) detail(retained, false);
    else {
      card.innerHTML = `<div class="terrain-point-empty"><span>${icon('target', 25)}</span><h3>${visible.length ? 'Select an observation' : 'No reading to inspect'}</h3><p>${visible.length ? 'Choose a point on the map or search the recorded readings below.' : 'Choose a different date, zone, or indicator.'}</p></div>`;
      updateSearch();
    }
  }

  function setDate(date) {
    if (!byDate.has(date)) return;
    state.date = date;
    dateSelect.value = date;
    range.value = String(dates.indexOf(date));
    if (redrawFrame) cancelAnimationFrame(redrawFrame);
    redrawFrame = requestAnimationFrame(() => { redrawFrame = 0; redraw(); });
  }

  function setExpanded(value) {
    frame.classList.toggle('is-expanded', value);
    frame.dataset.expanded = String(value);
    expand.innerHTML = icon(value ? 'close' : 'expand', 18);
    expand.title = value ? 'Close expanded map' : 'Expand map';
    expand.setAttribute('aria-label', expand.title);
    expand.setAttribute('aria-pressed', String(value));
    if (map) requestAnimationFrame(() => { if (!disposed) map.invalidateSize({pan: false}); });
  }

  function closeOnEscape(event) {
    if (event.key !== 'Escape' || frame.dataset.expanded !== 'true') return;
    event.preventDefault();
    setExpanded(false);
    expand.focus();
  }

  query('[data-terrain-parameter]').onchange = event => { state.parameter = event.currentTarget.value; redraw(); };
  query('[data-terrain-zone]').onchange = event => { state.zone = event.currentTarget.value; redraw(); fit(); };
  for (const button of host.querySelectorAll('[data-terrain-layer]')) button.onclick = () => layer(button.dataset.terrainLayer);
  dateSelect.onchange = () => setDate(dateSelect.value);
  range.oninput = () => setDate(dates[Number(range.value)]);
  query('[data-terrain-prev]').onclick = () => setDate(dates[dates.indexOf(state.date) - 1]);
  query('[data-terrain-next]').onclick = () => setDate(dates[dates.indexOf(state.date) + 1]);
  query('[data-terrain-fit]').onclick = fit;
  search.oninput = updateSearch;
  results.onclick = event => {
    const button = event.target.closest('[data-terrain-reading]');
    if (!button) return;
    const position = Number(button.dataset.terrainReading);
    detail(position, true);
    results.querySelector(`[data-terrain-reading="${position}"]`)?.focus();
  };
  expand.onclick = () => setExpanded(frame.dataset.expanded !== 'true');
  document.addEventListener('keydown', closeOnEscape);
  layer(state.layer);
  redraw();
  fit();

  const cleanup = () => {
    if (disposed) return;
    disposed = true;
    cancelAnimationFrame(redrawFrame);
    document.removeEventListener('keydown', closeOnEscape);
    resizeObserver?.disconnect();
    map?.remove();
  };
  if (ctx.signal?.aborted) cleanup();
  else ctx.signal?.addEventListener('abort', cleanup, {once: true});
  return cleanup;
}
