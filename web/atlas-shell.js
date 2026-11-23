(() => {
  'use strict';

  const footprintCache = new WeakMap();
  const counts = new Intl.NumberFormat('en-IN');
  const zoneColors = ['#79ebc9', '#59c9e0', '#eac67a', '#a6a9fa', '#64d3a2', '#edab9e'];
  const sectionNames = {
    overview: 'Monitor', map: 'Explore', zones: 'Explore', alerts: 'Monitor',
    forecasts: 'Forecast', validation: 'Forecast', evidence: 'Reference', data: 'Reference'
  };

  function footprint(rows, lastDate) {
    const cached = footprintCache.get(rows);
    if (cached?.lastDate === lastDate) return cached;
    const points = [];
    const zones = new Set();
    for (const row of rows) {
      if (row.zone_id != null) zones.add(String(row.zone_id));
      if (String(row.date).slice(0, 10) !== lastDate || row.lat == null || row.lon == null) continue;
      if (!Number.isFinite(+row.lat) || !Number.isFinite(+row.lon)) continue;
      points.push({lat: +row.lat, lon: +row.lon, zone: String(row.zone_id ?? '')});
    }
    const result = {lastDate, points, zones: [...zones].sort()};
    footprintCache.set(rows, result);
    return result;
  }

  function plotAtlas(rows, latest, escape, icon) {
    const {points, zones} = footprint(rows, latest);
    if (!points.length) return '';
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const {lat, lon} of points) {
      xMin = Math.min(xMin, lon);
      xMax = Math.max(xMax, lon);
      yMin = Math.min(yMin, lat);
      yMax = Math.max(yMax, lat);
    }
    const midLat = (yMin + yMax) / 2;
    const longitudeScale = Math.cos(midLat * Math.PI / 180);
    const scale = Math.min(312 / (Math.max((xMax - xMin) * longitudeScale, .00001)), 160 / Math.max(yMax - yMin, .00001));
    const zoneIndex = new Map(zones.map((zone, index) => [zone, index]));
    const sampled = points.length > 650 ? points.filter((_, index) => index % Math.ceil(points.length / 650) === 0) : points;
    const dots = sampled.map((point, index) => {
      const x = (214 + (point.lon - (xMin + xMax) / 2) * longitudeScale * scale).toFixed(2);
      const y = (112 - (point.lat - midLat) * scale).toFixed(2);
      const color = zoneColors[(zoneIndex.get(point.zone) ?? 0) % zoneColors.length];
      return `<circle cx="${x}" cy="${y}" r="${index % 6 === 0 ? 2.6 : 1.7}" fill="${color}" opacity="${index % 6 === 0 ? 1 : .72}"/>`;
    }).join('');
    const grids = Array.from({length: 9}, (_, index) => `<path d="M${33 + index * 45} 9v207"/>`).join('') +
      Array.from({length: 6}, (_, index) => `<path d="M16 ${18 + index * 36}h396"/>`).join('');
    const bands = [0, 1, 2, 3, 4, 5].map(index => {
      const size = index * 13;
      return `<path d="M${284 + size} -45c-35 58-177 59-178 140s148 38 151 115-131 60-145 102" transform="translate(${size / 2} ${size / 5})"/>`;
    }).join('');
    return `<figure class="atlas-plot">
      <div class="atlas-plot-top"><span>${icon('globe')} Observation footprint</span><span class="atlas-plot-label">LATEST RECORD</span></div>
      <svg class="atlas-coordinates" viewBox="0 0 428 224" aria-hidden="true">
        <g stroke="#6eaba9" stroke-width="0.7" opacity=".2">${grids}</g>
        <g stroke="#53bfa9" stroke-width="1" fill="none" opacity=".16">${bands}</g>
        <path d="M26 31V18h13m363 13V18h-13M26 194v13h13m363-13v13h-13" fill="none" stroke="#86caba" stroke-width="1"/>
        <text x="39" y="42" fill="#9bd6ca" font-size="11" font-weight="650">N</text>
        <path d="m41 64 5-14 5 14-5-4z" fill="#8cdecb"/>
        ${dots}
      </svg>
      <figcaption><span><i aria-hidden="true"></i> ${escape(counts.format(points.length))} mapped records</span><span>${sampled.length < points.length ? 'Sample of locations' : 'Points coloured by zone'}</span></figcaption>
    </figure>`;
  }

  function pageHeader({page, label, title, description, rows, latest, report, formatDate, escape, icon}) {
    if (page !== 'assessment') return `<div class="page-heading atlas-page-heading">
      <div class="atlas-page-icon">${icon(({overview:'grid',map:'globe',zones:'layers',alerts:'alert',forecasts:'chart',validation:'check',evidence:'book',data:'database'})[page])}</div>
      <div class="page-heading-copy"><div class="eyebrow">${escape(sectionNames[page] ?? 'Workspace')} <span aria-hidden="true">/</span> ${escape(label)}</div>
      <h1>${escape(title)}</h1><p>${escape(description)}</p></div>
    </div>`;
    const plot = plotAtlas(rows, latest, escape, icon);
    const summary = footprint(rows, latest);
    const recordedDates = Number(report.periods);
    const statistics = [[formatDate(latest), 'Latest recorded date']];
    if (Number.isFinite(recordedDates) && recordedDates > 0) statistics.push([counts.format(recordedDates), 'Observation dates']);
    if (summary.zones.length) statistics.push([counts.format(summary.zones.length), 'Monitoring zones']);
    return `<section class="atlas-hero ${plot ? 'has-footprint' : ''}" aria-labelledby="atlas-title">
      <div class="atlas-hero-copy">
        <div class="atlas-kicker"><i aria-hidden="true"></i> SATELLITE WATER MONITORING</div>
        <h1 id="atlas-title">Water condition</h1>
        <p>See what the latest satellite readings suggest about the lake, which uses need confirmation, and which sources support the precautions.</p>
        <div class="atlas-actions">
          <a href="#atlas-condition-report" class="btn atlas-primary" data-atlas-report>${icon('check')} View condition report</a>
          <button class="btn atlas-secondary" type="button" data-open-page="map">${icon('globe')} Explore map</button>
        </div>
        <dl class="atlas-stats">${statistics.map(([value, key]) => `<div><dt>${escape(key)}</dt><dd>${escape(value)}</dd></div>`).join('')}</dl>
      </div>
      ${plot}
    </section>`;
  }

  window.AquaAtlas = {pageHeader};
})();
