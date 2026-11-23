const shapes = {
  drop: 'M12 2C9 7 5 10.8 5 15a7 7 0 0 0 14 0c0-4.2-4-8-7-13ZM8.6 15.1a3.5 3.5 0 0 0 3.4 3.4',
  activity: 'M2 12h5l3-7 4 14 3-7h5', grid:'M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z',
  map: 'm3 6 6-3 6 3 6-3v15l-6 3-6-3-6 3zM9 3v15m6-12v15',
  chart: 'M3 3v18h18M7 15l4-5 4 2 5-7', target: 'M22 12a10 10 0 1 1-20 0 10 10 0 1 1 20 0ZM17 12a5 5 0 1 1-10 0 5 5 0 1 1 10 0ZM12 12h.01',
  layers: 'm12 3 10 6-10 6L2 9l10-6ZM2 13l10 6 10-6M2 17l10 6 10-6', pin: 'M20 10c0 6-8 12-8 12S4 16 4 10a8 8 0 0 1 16 0ZM12 10h.01',
  book: 'M12 6c-4-3-7-3-10-2v15c4-1 7 0 10 2 3-2 6-3 10-2V4c-3-1-6-1-10 2Zm0 0v15',
  database: 'M21 5c0 2-4 3-9 3S3 7 3 5s4-3 9-3 9 1 9 3ZM3 5v14c0 2 4 3 9 3s9-1 9-3V5M3 12c0 2 4 3 9 3s9-1 9-3',
  upload: 'M12 16V3m-5 5 5-5 5 5M3 15v6h18v-6', download: 'M12 3v13m-5-5 5 5 5-5M3 17v4h18v-4',
  calendar: 'M3 5h18v16H3zM7 3v4m10-4v4M3 10h18', clock: 'M12 8v5l3 2M22 12a10 10 0 1 1-20 0 10 10 0 1 1 20 0',
  arrow: 'M4 12h16m-6-6 6 6-6 6', chevron: 'm9 5 7 7-7 7', down: 'm6 9 6 6 6-6',
  close: 'm6 6 12 12M6 18 18 6', check: 'm5 12 4 4L19 6',
  alert: 'm12 3 10 18H2zM12 9v5M12 17v.1', info:'M12 11v6M12 7v.1M22 12a10 10 0 1 1-20 0 10 10 0 1 1 20 0',
  search: 'M21 21l-5-5M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0', external: 'M14 3h7v7m0-7L10 14M10 3H3v18h18v-7',
  menu: 'M3 6h18M3 12h18M3 18h18', filter:'M3 5h18M6 12h12M9 19h6',
  globe:'M22 12a10 10 0 1 1-20 0 10 10 0 1 1 20 0M2 12h20M12 2c6 6 6 14 0 20-6-6-6-14 0-20',
  refresh:'M20 7v5h-5M4 17v-5h5M5 7a8 8 0 0 1 14-2l1 7M4 12l1 7a8 8 0 0 0 14-2',
  spark:'m12 2 3 7 7 3-7 3-3 7-3-7-7-3 7-3z', expand:'M3 8V3h5m8 0h5v5m0 8v5h-5m-8 0H3v-5',
  waves: 'M2 8c3 0 3-2 5-2s3 2 5 2 3-2 5-2 3 2 5 2M2 14c3 0 3-2 5-2s3 2 5 2 3-2 5-2 3 2 5 2M2 20c3 0 3-2 5-2s3 2 5 2 3-2 5-2 3 2 5 2',
};

export const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, character => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
})[character]);

export const icon = (name, size = 20) => `<svg class="station-icon" width="${Number(size) || 20}" height="${Number(size) || 20}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linejoin="round" stroke-linecap="round" aria-hidden="true"><path d="${shapes[name] || shapes.info}"/></svg>`;

export function formatNumber(value, digits = 2) {
  if (value == null || value === '' || !Number.isFinite(+value)) return '—';
  return new Intl.NumberFormat('en-IN', {maximumFractionDigits: digits, minimumFractionDigits: digits}).format(+value);
}

export function formatDate(value, options = {}) {
  if (!value) return '—';
  const parsed = new Date(`${String(value).slice(0, 10)}T12:00:00Z`);
  if (Number.isNaN(parsed.valueOf())) return '—';
  const fields = options === true ? {month: 'short', day: 'numeric'} : {day: 'numeric', month: 'short', year: 'numeric', ...options};
  return new Intl.DateTimeFormat('en-GB', {...fields, timeZone: 'UTC'}).format(parsed);
}

export function safeUrl(value) {
  try { const url = new URL(value); return url.protocol === 'https:' ? url.href : ''; }
  catch { return ''; }
}

export function download(content, filename, mime = 'text/plain;charset=utf-8') {
  const artifact = new Blob([content], {type: mime});
  const url = URL.createObjectURL(artifact);
  const anchor = Object.assign(document.createElement('a'), {href: url, download: filename});
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function toCsv(rows, keys = Object.keys(rows[0] || {})) {
  const quote = value => {
    const raw = String(value ?? '');
    const protectedText = typeof value === 'string' && /^[\s]*[=+\-@]/.test(raw) ? `'${raw}` : raw;
    return `"${protectedText.replaceAll('"', '""')}"`;
  };
  return [keys.map(quote).join(','), ...rows.map(row => keys.map(key => quote(row[key])).join(','))].join('\r\n');
}

export const parameterNames = {ndwi: 'Water index (NDWI)', turbidity_proxy: 'Turbidity proxy', chlorophyll_proxy: 'Chlorophyll proxy'};
