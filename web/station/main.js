import {escapeHtml as html, formatDate, formatNumber, icon} from './utils.js';
import {renderRecords, renderSources, renderAlerts} from './records.js';
import {savedOutlook} from './outlook-reference.js';

const workspace = document.querySelector('#workspace');
const content = document.querySelector('#route-content');
const state = {job: null, trainingPending: false};
const routes = new Set(['assessment', 'overview', 'map', 'zones', 'forecasts', 'validation', 'evidence', 'alerts', 'data']);
const routeNames = {assessment:'Monitor', overview:'Lake overview', map:'Observation map', zones:'Zone details', forecasts:'Forecast', validation:'Accuracy', evidence:'Sources', alerts:'Change alerts', data:'Data'};
const defaultDataset = 'modis';
let data = null, mode = 'api', activeRoute = null, activeView = null, toastTime;
let routeSignal, pollTime;

const currentRoute = () => {
  const hash = location.hash.slice(1);
  return routes.has(hash) ? hash : 'assessment';
};

function notify(message, kind = 'info') {
  const node = document.querySelector('#notification');
  clearTimeout(toastTime);
  node.textContent = message;
  node.dataset.kind = kind;
  node.hidden = false;
  toastTime = setTimeout(() => { node.hidden = true; }, 6500);
}

async function request(path, {body, method = body ? 'POST' : 'GET', signal, timeout = 45000, ...options} = {}) {
  const controller = new AbortController();
  const stop = () => controller.abort(signal?.reason);
  if (signal?.aborted) stop();
  else signal?.addEventListener('abort', stop, {once: true});
  const timer = setTimeout(() => controller.abort(new Error('The local engine took too long to respond.')), timeout);
  try {
    const response = await fetch(path, {
      method, ...options, signal: controller.signal,
      headers: body ? {'Content-Type': 'application/json', ...options.headers} : options.headers,
      body: body && typeof body !== 'string' ? JSON.stringify(body) : body,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `The request failed (${response.status}).`);
    return result;
  } catch (error) {
    if (controller.signal.aborted && controller.signal.reason?.message) throw controller.signal.reason;
    throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', stop);
  }
}

const datasetID = () => typeof data?.dataset === 'string' ? data.dataset : defaultDataset;
const bootstrap = (dataset, evidence = '') => request(`/api/bootstrap?${new URLSearchParams({dataset, ...(evidence ? {evidence_id: evidence} : {})})}`, {timeout:60000});

function loading(title, description) {
  return `<section class="station-loading" aria-live="polite"><span class="spinner"></span><h1>${html(title)}</h1><p>${html(description)}</p></section>`;
}

function showError(host, error) {
  host.innerHTML = `<section class="station-empty" role="alert">${icon('alert',26)}<h2>We couldn’t open this view</h2><p>${html(error.message || error)}</p><button class="station-button" type="button" data-retry>Try again</button></section>`;
  host.querySelector('[data-retry]').onclick = () => data ? render() : boot();
}

function displayContext() {
  const report = data?.report || {};
  document.querySelector('#dataset-label').textContent = datasetID() === defaultDataset ? 'MODIS lake observations' : (report.lake_id || 'Uploaded lake');
  document.querySelector('#dataset-period').textContent = report.start && report.end ? `${formatDate(report.start)} – ${formatDate(report.end)}` : 'Observation history';
  const status = document.querySelector('#workspace-status');
  status.textContent = mode === 'api' ? 'Local workspace' : 'Saved dataset';
  status.dataset.mode = mode;
}

function navigate(route) {
  if (!routes.has(route)) return;
  if (route === currentRoute()) { window.scrollTo({top:0}); if (activeRoute !== route) render(); }
  else location.hash = route;
}

async function renderMonitor(host, ctx) {
  return renderModule(host, ctx, () => import('./quality.js'), 'renderQuality');
}

async function renderModule(host, ctx, load, view) {
  host.innerHTML = loading(`Opening ${routeNames[ctx.route].toLowerCase()}`, 'Preparing this view.');
  const module = await load();
  if (!ctx.signal.aborted) return module[view](host, ctx);
}

async function render() {
  if (!data) return;
  const route = currentRoute();
  routeSignal?.abort();
  if (typeof activeView === 'function') activeView();
  activeView = null;
  activeRoute = route;
  const controller = new AbortController();
  routeSignal = controller;
  const navRoute = ['assessment','overview','alerts'].includes(route) ? 'assessment' : route === 'zones' ? 'map' : route;
  document.title = `${routeNames[route]} · AquaLens`;
  document.querySelectorAll('[data-route]').forEach(link => {
    if (link.dataset.route === navRoute) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
  const host = document.createElement('div');
  host.className = `station-route route-${route}`;
  content.replaceChildren(host);
  const geography = () => import('./geography.js');
  const outlook = () => import('./outlook.js');
  const views = {assessment:renderMonitor, overview:renderMonitor,
    map:(node, view) => renderModule(node, view, geography, 'renderGeography'),
    zones:(node, view) => renderModule(node, view, geography, 'renderGeography'),
    forecasts:(node, view) => renderModule(node, view, outlook, 'renderForecast'),
    validation:(node, view) => renderModule(node, view, outlook, 'renderAccuracy'),
    evidence:renderSources, data:renderRecords, alerts:renderAlerts};
  const activeData = data;
  const ctx = {data, mode, request, navigate, state, route, signal:controller.signal, notify,
    savedOutlook: (kind, selection) => savedOutlook(activeData, request, kind, selection),
    useBundled, train: startTraining, upload: () => document.querySelector('#upload-file').click(),
    uploadSources: () => document.querySelector('#source-file').click()};
  try {
    const cleanup = await views[route](host, ctx);
    if (typeof cleanup === 'function') {
      if (controller.signal.aborted) cleanup();
      else activeView = cleanup;
    }
  } catch (error) {
    if (!controller.signal.aborted && host.isConnected) showError(host, error);
  }
}

async function boot() {
  content.innerHTML = loading('Opening your workspace', 'Loading observations, forecasts, and linked evidence.');
  try {
    data = await bootstrap(defaultDataset);
    mode = 'api';
  } catch {
    try {
      data = await request('./demo-state.json', {timeout:45000});
      mode = 'snapshot';
    } catch (error) { showError(content, error); return; }
  }
  displayContext();
  render();
}

async function useBundled() {
  try {
    const bundled = mode === 'api' ? await bootstrap(defaultDataset) : await request('./demo-state.json');
    replaceData(bundled);
    notify('Opened the supplied MODIS dataset.');
  } catch (error) { notify(error.message, 'error'); }
}

function replaceData(next, {keepSelections = false} = {}) {
  if (!keepSelections) for (const key of Object.keys(state)) if (!['job','trainingPending'].includes(key)) delete state[key];
  data = next;
  displayContext();
  render();
}

document.querySelector('#upload-file').addEventListener('change', async event => {
  const file = event.target.files[0];
  event.target.value = '';
  if (!file) return;
  if (mode !== 'api') { notify('Start the local engine before uploading a CSV.', 'error'); return; }
  if (file.size > 11 * 1024 * 1024) { notify('Choose a CSV smaller than 11 MB.', 'error'); return; }
  const buttons = [...document.querySelectorAll('[data-upload]')];
  buttons.forEach(button => { button.disabled = true; });
  notify('Importing observations…');
  try {
    const next = await request('/api/upload', {body:{filename:file.name, csv:await file.text()}, timeout:120000});
    replaceData(next);
    navigate('assessment');
    notify(`Imported ${formatNumber(next.report?.rows,0)} observations.`, 'success');
  } catch (error) { notify(error.message || 'Upload failed.', 'error'); }
  finally { buttons.forEach(button => { button.disabled = false; }); }
});

document.querySelector('#source-file').addEventListener('change', async event => {
  const file = event.target.files[0];
  event.target.value = '';
  if (!file) return;
  if (mode !== 'api') { notify('Start the local engine to add a source collection.', 'error'); return; }
  if (file.size > 2 * 1024 * 1024) { notify('Choose a JSON file smaller than 2 MB.', 'error'); return; }
  try {
    const records = JSON.parse(await file.text());
    const {evidence_id} = await request('/api/knowledge', {body:{records}});
    replaceData(await bootstrap(datasetID(), evidence_id), {keepSelections:true});
    notify('Source collection imported.', 'success');
  } catch (error) { notify(error.message || 'Import failed.', 'error'); }
});

function refreshJobPanel() {
  const container = document.querySelector('[data-training-status]');
  if (!container) return;
  const job = state.job;
  if (!job) return;
  container.hidden = false;
  container.querySelector('strong').textContent = job.status === 'complete' ? 'Training complete' : job.status === 'failed' ? 'Training failed' : 'Training in progress';
  container.querySelector('progress').value = job.progress || 0;
  container.querySelector('p').textContent = job.error || job.message || '';
  const submit = document.querySelector('[data-training-form] button[type=submit]');
  if (submit) submit.disabled = ['queued','running'].includes(job.status);
}

async function pollJob(id, failures = 0) {
  try {
    const job = await request(`/api/jobs/${encodeURIComponent(id)}`);
    state.job = job;
    refreshJobPanel();
    if (job.status === 'complete') {
      notify('DET training is complete.', 'success');
      if (job.dataset === datasetID()) replaceData(await bootstrap(datasetID(), data.evidence_id), {keepSelections:true});
      return;
    }
    if (job.status === 'failed') { notify(job.error || 'Training failed.', 'error'); return; }
    pollTime = setTimeout(() => pollJob(id), 1800);
  } catch (error) {
    if (failures < 2) pollTime = setTimeout(() => pollJob(id, failures + 1), 2500);
    else notify(error.message, 'error');
  }
}

async function startTraining(epochs) {
  if (mode !== 'api' || state.trainingPending || ['queued','running'].includes(state.job?.status)) return;
  state.trainingPending = true;
  try {
    state.job = await request('/api/train', {body:{dataset:datasetID(), epochs}});
    refreshJobPanel();
    clearTimeout(pollTime);
    pollJob(state.job.id);
  } catch (error) { notify(error.message, 'error'); }
  finally { state.trainingPending = false; }
}

document.querySelectorAll('[data-symbol]').forEach(node => { node.innerHTML = icon(node.dataset.symbol, 17); });
document.querySelector('.header-upload').onclick = () => document.querySelector('#upload-file').click();
document.querySelector('.skip-link').onclick = event => { event.preventDefault(); workspace.focus(); };
document.querySelectorAll('[data-route]').forEach(link => link.addEventListener('click', event => {
  if (event.defaultPrevented || event.ctrlKey || event.shiftKey || event.altKey || event.metaKey) return;
  if (link.dataset.route === currentRoute() && activeRoute === currentRoute()) {
    event.preventDefault();
    window.scrollTo({top:0});
  }
}));
window.addEventListener('hashchange', () => {
  if (currentRoute() !== activeRoute) { window.scrollTo({top:0}); render(); }
});

boot();
