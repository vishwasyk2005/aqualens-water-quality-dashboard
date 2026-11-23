const assetPaths = {
  forecast: './station/forecast-reference.json',
  backtest: './station/backtest-reference.json',
};
const requests = new Map();

export async function savedOutlook(data, request, kind, {model, days, zone} = {}) {
  const benchmark = data?.model_benchmark;
  if (model !== 'auto' || data?.dataset !== 'modis' || !benchmark?.saved_history ||
    !benchmark.fingerprint || !assetPaths[kind]) return null;
  if (kind === 'forecast' && (!Number.isInteger(days) || days < 1 || days > 32)) return null;

  if (!requests.has(kind)) {
    requests.set(kind, request(assetPaths[kind], {timeout: 7000}).catch(() => {
      requests.delete(kind);
      return null;
    }));
  }
  const asset = await requests.get(kind);
  if (asset?.schema !== 1 || asset.kind !== kind || asset.dataset !== data.dataset ||
    asset.fingerprint !== benchmark.fingerprint || asset.selected_model !== benchmark.selected_model) return null;
  const payload = asset.payload;
  if (!Array.isArray(payload?.rows)) return null;
  if (kind === 'forecast' && days > payload.maximum_days) return null;
  const rows = kind === 'backtest' && zone
    ? payload.rows.filter(row => String(row.zone_id) === zone)
    : kind === 'forecast' ? payload.rows.filter(row => row.horizon_days <= days) : payload.rows;
  if (!rows.length) return null;
  return {...payload, rows, metadata: {...payload.metadata,
    display_source: kind === 'forecast' ? 'Saved forecast for the supplied MODIS data' : 'Frozen historical benchmark'}};
}
