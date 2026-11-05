"""Validated causal CSV ingestion. Proxy columns are sensor-specific indices."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd

PARAMETERS = ['ndwi', 'turbidity_proxy', 'chlorophyll_proxy']
DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
ZONE_NAMES = ['Northern inlet', 'North shore', 'Central west', 'Central east', 'Southern shore', 'Outlet']

def _bool(s):
    return s.astype(str).str.strip().str.lower().isin(['true', '1', 'yes', '1.0'])

def _coordinates(value):
    try:
        item = json.loads(value) if isinstance(value, str) else value
        if item.get('type') != 'Point':
            return np.nan, np.nan
        lon, lat = item['coordinates'][:2]
        return float(lon), float(lat)
    except (ValueError, TypeError, KeyError, AttributeError):
        return np.nan, np.nan

def _cell_id(lat, lon):
    return 'cell_' + hashlib.sha256(f'{lat:.6f},{lon:.6f}'.encode()).hexdigest()[:12]

def load_data(source):
    """Accept path, bytes or file; return (DataFrame, JSON-serializable report).

    Missing values may use a preceding value within the SAME cell only when
    at most 14 days old and at most two consecutive rows are missing. Leading
    gaps and long gaps stay missing. Original anomalies are never clipped.
    """
    try:
        df = pd.read_csv(io.BytesIO(source) if isinstance(source, bytes) else source)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise ValueError(f'Cannot read CSV: {exc}') from exc
    if df.empty:
        raise ValueError('CSV contains no observations.')
    input_rows = len(df)
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any():
        raise ValueError('CSV contains duplicate column names.')
    legacy = 'system:index' in df and '.geo' in df
    aliases = {'NDWI':'ndwi', 'Turbidity':'turbidity_proxy', 'Chlorophyll':'chlorophyll_proxy', 'latitude':'lat', 'longitude':'lon', 'waterbody_id':'lake_id', 'temperature_c':'temp_c'}
    df = df.rename(columns={k:v for k,v in aliases.items() if k in df and v not in df})
    if legacy:
        dates = df['system:index'].astype(str).str.extract(r'(\d{4}_\d{2}_\d{2})', expand=False)
        df['date'] = pd.to_datetime(dates, format='%Y_%m_%d', errors='coerce')
        coords = df['.geo'].map(_coordinates)
        df['lon'], df['lat'] = zip(*coords)
    required = ['date', 'lat', 'lon'] + PARAMETERS
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    df['date'] = pd.to_datetime(df.date, errors='coerce', utc=True).dt.tz_localize(None).dt.normalize()
    for c in ['lat', 'lon'] + PARAMETERS:
        df[c] = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    valid = df.date.notna() & df.lat.between(-90,90) & df.lon.between(-180,180)
    invalid_rows = int((~valid).sum())
    df = df.loc[valid].copy()
    if df.empty:
        raise ValueError('No rows have valid dates and coordinates (lat -90..90, lon -180..180).')
    for c in PARAMETERS:
        if not df[c].notna().any():
            raise ValueError(f'Parameter {c} has no valid numeric values.')
    if 'lake_id' not in df:
        df['lake_id'] = 'legacy_lake' if legacy else 'uploaded_lake'
    for c in ['lake_id', 'zone_id', 'cell_id']:
        if c in df:
            if df[c].isna().any() or df[c].astype(str).str.strip().eq('').any():
                raise ValueError(f'{c} contains empty identifiers.')
            df[c] = df[c].astype(str)
    if df.lake_id.nunique() != 1:
        raise ValueError('Upload one lake at a time; multiple lake_id values found.')
    if 'cell_id' not in df:
        df['cell_id'] = [_cell_id(a,b) for a,b in zip(df.lat,df.lon)]
    inferred_zones = 'zone_id' not in df
    if inferred_zones:
        lat_span = max(float(df.lat.max()-df.lat.min()),1e-9)
        lon_span = max(float(df.lon.max()-df.lon.min()),1e-9)
        a = np.minimum(((df.lat-df.lat.min())/lat_span*3).astype(int),2)
        b = np.minimum(((df.lon-df.lon.min())/lon_span*2).astype(int),1)
        df['zone_id'] = 'Z' + (a*2+b+1).astype(str)
    if 'zone_name' not in df:
        df['zone_name'] = 'Spatial zone ' + df.zone_id
    if (df.groupby('cell_id').zone_id.nunique()>1).any():
        raise ValueError('A cell_id belongs to multiple zones; use a stable hierarchy.')
    movement = df.groupby('cell_id')[['lat','lon']].agg(lambda x:x.max()-x.min())
    if (movement>0.0001).any().any():
        raise ValueError('A cell_id has changing coordinates; cells need stable locations.')
    duplicate_rows = int(df.duplicated(['date','cell_id']).sum())
    df = df.drop_duplicates(['date','cell_id'],keep='first').sort_values(['cell_id','date']).reset_index(drop=True)
    for c in ['rainfall_mm','temp_c','valid_pixel_fraction']:
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    bad_coverage = df.valid_pixel_fraction.notna() & ~df.valid_pixel_fraction.between(0,1)
    df.loc[bad_coverage,'valid_pixel_fraction'] = np.nan
    source_observed = _bool(df.is_observed) if 'is_observed' in df else pd.Series(True,index=df.index)
    for c in PARAMETERS:
        raw = c+'_raw'
        df[raw] = pd.to_numeric(df[raw],errors='coerce').replace([np.inf,-np.inf],np.nan) if raw in df else df[c]
    invalid_ndwi = df.ndwi_raw.notna() & ~df.ndwi_raw.between(-1,1)
    df['ndwi_domain_invalid'] = invalid_ndwi
    df.loc[invalid_ndwi,'ndwi'] = np.nan
    for c in PARAMETERS:
        flag = c+'_was_missing'
        df[flag] = df[c].isna() | (_bool(df[flag]) if flag in df else False)
        original_flag = _bool(df[c+'_is_observed']) if c+'_is_observed' in df else pd.Series(True,index=df.index)
        observed = source_observed & original_flag & df[c].notna() & ~df[flag]
        df[c+'_is_observed'] = observed
        valid_values = df[c].where(observed)
        previous_date = df.date.where(observed).groupby(df.cell_id).ffill()
        previous_value = valid_values.groupby(df.cell_id).ffill(limit=2)
        age = (df.date-previous_date).dt.days
        fill_allowed = age.le(14)
        if c=='ndwi': fill_allowed &= ~invalid_ndwi
        df[c] = df[c].where(df[c].notna(),previous_value.where(fill_allowed))
    df['is_observed'] = df[[c+'_is_observed' for c in PARAMETERS]].any(axis=1)
    df['proxy_negative_flag'] = df.turbidity_proxy_raw.lt(0) | df.chlorophyll_proxy_raw.lt(0)
    if 'sensor' not in df:
        df['sensor'] = 'MODIS (legacy; algorithm unverified)' if legacy else 'unspecified'
    if 'source_kind' not in df:
        df['source_kind'] = 'legacy_upload' if legacy else 'uploaded'
    if df.sensor.nunique()>1:
        raise ValueError('Multiple sensors found; upload one compatible sensor dataset at a time.')
    if 'algorithm_version' in df and df.algorithm_version.dropna().nunique()>1:
        raise ValueError('Multiple algorithm versions found; upload compatible values.')
    warnings = []
    if inferred_zones:
        warnings.append('Zones inferred from a 3x2 geographic grid; provide zone_id for meaningful boundaries.')
    if legacy:
        warnings.append('Legacy raw values retained; NDWI outside [-1,1] excluded from analysis. Source algorithms and units unverified; do not mix with Sentinel-2 proxies.')
    if invalid_ndwi.any():
        warnings.append(f'{int(invalid_ndwi.sum())} NDWI values outside [-1,1] excluded, with raw values preserved. They are not filled or capped.')
    if df.proxy_negative_flag.any():
        warnings.append('Negative turbidity/chlorophyll source values exist. Units and retrieval formulas need verification; values are retained as unverified proxies.')
    if invalid_rows:
        warnings.append(f'Dropped {invalid_rows} rows with invalid dates or coordinates.')
    if duplicate_rows:
        warnings.append(f'Removed {duplicate_rows} duplicate date/cell rows; first occurrence retained.')
    if bad_coverage.any():
        warnings.append('Invalid pixel coverage fractions were marked unknown.')
    if df.valid_pixel_fraction.isna().any():
        warnings.append('Some pixel coverage is unknown; unknown coverage is not treated as 100%.')
    if df[PARAMETERS].isna().any().any():
        warnings.append('Long or leading gaps remain missing; models require usable contiguous windows.')
    if (df.source_kind=='synthetic_demo').any():
        warnings.append('SYNTHETIC DEMO: generated observations, locations, weather and anomalies; not measured lake conditions.')
    report = dict(input_rows=int(input_rows),rows=int(len(df)),invalid_rows=invalid_rows,duplicate_rows=duplicate_rows,cells=int(df.cell_id.nunique()),zones=int(df.zone_id.nunique()),lake_id=str(df.lake_id.iloc[0]),source=str(df.source_kind.iloc[0]),start=df.date.min().strftime('%Y-%m-%d'),end=df.date.max().strftime('%Y-%m-%d'),periods=int(df.date.nunique()),missing_fraction=float(df[PARAMETERS].isna().mean().mean()),original_missing_fraction=float(df[[p+'_was_missing' for p in PARAMETERS]].mean().mean()),observed_fraction=float(df.is_observed.mean()),coverage=float(df.valid_pixel_fraction.mean()) if df.valid_pixel_fraction.notna().any() else None,warnings=warnings)
    report.update(ndwi_invalid_count=int(invalid_ndwi.sum()),proxy_negative_count=int(df.proxy_negative_flag.sum()),coverage_unknown_fraction=float(df.valid_pixel_fraction.isna().mean()),parameter_observed_fraction={c:float(df[c+'_is_observed'].mean()) for c in PARAMETERS},cadence_days=float(df.date.drop_duplicates().sort_values().diff().dt.days.median()) if df.date.nunique()>1 else None,units='Unverified source proxy scale; no physical concentration or safety claim')
    return df.sort_values(['date','zone_id','cell_id']).reset_index(drop=True),report

def load_demo():
    return load_data(DATA_DIR/'demo.csv')

def zones_summary(df):
    """Latest common-date means; stale cells are never silently reused."""
    latest = df.loc[df.date.eq(df.date.max())]
    latest=latest.copy()
    for c in PARAMETERS:
        flag=c+'_is_observed'
        if flag in latest: latest.loc[~latest[flag],c]=np.nan
    result = latest.groupby(['zone_id','zone_name'],as_index=False).agg(lat=('lat','mean'),lon=('lon','mean'),date=('date','max'),cells=('cell_id','nunique'),coverage=('valid_pixel_fraction','mean'),observations=('is_observed','sum'),**{p:(p,'mean') for p in PARAMETERS})
    result['expected_cells'] = result.zone_id.map(df.groupby('zone_id').cell_id.nunique())
    return result

def generate_demo():
    """Deterministic synthetic scenario; not a named real lake."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(end='2025-12-28',periods=156,freq='W-SUN')
    records=[]
    for t,date in enumerate(dates):
        season=np.sin(2*np.pi*(t-12)/52)
        rain=max(0,22+31*season+rng.normal(0,8))
        recent=max(0,(t-149)/6)
        for z in range(6):
            for c in range(4):
                localized=recent if z==0 else recent*.18 if z==1 else 0
                record=dict(date=date.strftime('%Y-%m-%d'),lake_id='demo_kerala_lake',zone_id=f'Z{z+1}',zone_name=ZONE_NAMES[z],cell_id=f'Z{z+1}-C{c+1}',lat=9.68-(z//2)*.025+(c//2)*.005,lon=76.39+(z%2)*.027+(c%2)*.005,ndwi=.42+.065*season+.015*(z-2)+rng.normal(0,.012)-localized*.14,turbidity_proxy=.23+.045*season+.012*z+rng.normal(0,.012)+localized*.37,chlorophyll_proxy=.16+.035*np.sin(2*np.pi*(t-19)/52)+.007*z+rng.normal(0,.009)+localized*.11,rainfall_mm=rain+(recent*65 if z<2 else 0),temp_c=28.2-1.1*season+rng.normal(0,.35),valid_pixel_fraction=float(rng.uniform(.73,.99)),is_observed=True,sensor='SYNTHETIC_S2_STYLE',source_kind='synthetic_demo',algorithm_version='synthetic-v1')
                if 25<t<140 and rng.random()<.017:
                    record.update(ndwi=np.nan,turbidity_proxy=np.nan,chlorophyll_proxy=np.nan,is_observed=False,valid_pixel_fraction=.12)
                records.append(record)
    metadata=dict(title='AquaLens synthetic Kerala lake scenario',synthetic=True,seed=42,periods=156,frequency='weekly Sunday',cells=24,zones=6,start=dates.min().strftime('%Y-%m-%d'),end=dates.max().strftime('%Y-%m-%d'),description='Generated sensor-style indices and weather near Kerala. Not measurements of a real lake. A northern-inlet runoff-like anomaly is injected in the final six weeks.',units={p:'dimensionless synthetic index' for p in PARAMETERS},notes=['Synthetic coordinates are for interface demonstration only.','No operational safety or compliance thresholds implied.','Synthetic accuracy scores are not evidence of real-world performance.'])
    return pd.DataFrame(records),metadata

if __name__=='__main__':
    DATA_DIR.mkdir(exist_ok=True,parents=True)
    demo,metadata=generate_demo()
    demo.to_csv(DATA_DIR/'demo.csv',index=False,float_format='%.6f')
    (DATA_DIR/'demo_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(load_demo()[1],indent=2))
