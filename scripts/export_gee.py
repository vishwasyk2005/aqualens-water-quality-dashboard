"""Export weekly Sentinel-2 proxy measurements to Google Drive.

Requires earthengine-api and an authenticated Google Earth Engine project.
No import-time authentication, network calls, or exports. Values are
DIMENSIONLESS PROXIES; no claim of calibrated NTU or chlorophyll mg/L.
Each weekly row is dated at its last included calendar day, not the start of
the composite. observation_start and observation_end_exclusive preserve the
source interval. Allow for source processing latency in operational forecasts.

Sources:
https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
https://developers.google.com/earth-engine/apidocs/ee-image-reduceregions
https://developers.google.com/earth-engine/apidocs/export-table-todrive
"""
from __future__ import annotations
import argparse
import json
from datetime import date
from pathlib import Path

BANDS = ['ndwi','turbidity_proxy','chlorophyll_proxy']

def arguments():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',required=True,help='Your enabled Earth Engine Cloud project ID')
    parser.add_argument('--area',required=True,type=Path,help='GeoJSON Polygon/MultiPolygon, Feature, or FeatureCollection')
    parser.add_argument('--zones',type=Path,help='Optional GeoJSON FeatureCollection with zone_id and zone_name properties')
    parser.add_argument('--lake-id',default='study_lake')
    parser.add_argument('--start',required=True,type=date.fromisoformat)
    parser.add_argument('--end',required=True,type=date.fromisoformat,help='Exclusive end date')
    parser.add_argument('--grid-m',default=500,type=int,help='Cell edge in meters, default 500')
    parser.add_argument('--crs',default='EPSG:32643',help='Metric projected CRS; default UTM 43N for Kerala')
    parser.add_argument('--scale',default=20,type=int,help='Reduction scale in meters')
    parser.add_argument('--water-ndwi-min',default=-0.1,type=float,help='Uncalibrated water screen; tune with local reference data')
    parser.add_argument('--folder',default='AquaLens_exports')
    parser.add_argument('--authenticate',action='store_true',help='Run interactive local Google authentication first')
    return parser.parse_args()

def build_export(ee,args):
    """Build the table without starting a task; ee must already be initialized."""
    raw=json.loads(args.area.read_text(encoding='utf-8-sig'))
    if raw.get('type')=='FeatureCollection':
        area=ee.FeatureCollection(raw).geometry()
    elif raw.get('type')=='Feature':
        area=ee.Geometry(raw['geometry'])
    else:
        area=ee.Geometry(raw)
    projection=ee.Projection(args.crs)
    bounds=area.bounds(1,projection).coordinates().get(0)
    lower=ee.List(ee.List(bounds).get(0))
    upper=ee.List(ee.List(bounds).get(2))
    xmin,ymin=ee.Number(lower.get(0)),ee.Number(lower.get(1))
    xmax,ymax=ee.Number(upper.get(0)),ee.Number(upper.get(1))
    zone_features=None
    if args.zones:
        zone_raw=json.loads(args.zones.read_text(encoding='utf-8-sig'))
        if zone_raw.get('type')!='FeatureCollection' or not zone_raw.get('features'):
            raise ValueError('--zones must be a nonempty GeoJSON FeatureCollection.')
        for feature in zone_raw['features']:
            if not feature.get('properties',{}).get('zone_id') or not feature.get('properties',{}).get('zone_name'):
                raise ValueError('Every zone requires zone_id and zone_name properties.')
        zone_features=ee.FeatureCollection(zone_raw)
    def identify(feature):
        feature=ee.Feature(feature)
        geometry=feature.geometry().intersection(area,1)
        center=geometry.centroid(1)
        metric=center.transform(projection,1).coordinates()
        x,y=ee.Number(metric.get(0)),ee.Number(metric.get(1))
        coords=center.transform('EPSG:4326',1).coordinates()
        fallback=ee.String('Z').cat(y.subtract(ymin).divide(ymax.subtract(ymin)).multiply(3).floor().min(2).multiply(2).add(x.subtract(xmin).divide(xmax.subtract(xmin)).multiply(2).floor().min(1)).add(1).format('%d'))
        zone_id=fallback
        zone_name=ee.String('Spatial zone ').cat(fallback)
        if zone_features is not None:
            matches=zone_features.filterBounds(center)
            zone_id=ee.Algorithms.If(matches.size().gt(0),ee.Feature(matches.first()).get('zone_id'),'unassigned')
            zone_name=ee.Algorithms.If(matches.size().gt(0),ee.Feature(matches.first()).get('zone_name'),'Unassigned')
        # Grid geometry, not clipped centroid, defines the stable cell identifier.
        grid_xy=feature.geometry().centroid(1).transform(projection,1).coordinates()
        cell_id=ee.String('cell_').cat(ee.Number(grid_xy.get(0)).round().format('%d')).cat('_').cat(ee.Number(grid_xy.get(1)).round().format('%d'))
        return ee.Feature(geometry,dict(cell_id=cell_id,zone_id=zone_id,zone_name=zone_name,lat=coords.get(1),lon=coords.get(0),lake_id=args.lake_id))
    grid=area.coveringGrid(projection,args.grid_m).map(identify)
    def indices(image):
        scl=image.select('SCL')
        cloud_ok=scl.neq(0).And(scl.neq(1)).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        reflectance=image.select(['B3','B4','B5','B8']).multiply(.0001).updateMask(cloud_ok)
        ndwi=reflectance.expression('(g-n)/(g+n+1e-6)',dict(g=reflectance.select('B3'),n=reflectance.select('B8'))).rename('ndwi')
        water=ndwi.gt(args.water_ndwi_min)
        # Red/green ratio and NDCI are proxies; keep their algorithm names/version.
        turbidity=reflectance.expression('r/(g+1e-6)',dict(r=reflectance.select('B4'),g=reflectance.select('B3'))).rename('turbidity_proxy')
        chlorophyll=reflectance.expression('(re-r)/(re+r+1e-6)',dict(re=reflectance.select('B5'),r=reflectance.select('B4'))).rename('chlorophyll_proxy')
        return ndwi.addBands(turbidity).addBands(chlorophyll).updateMask(water).copyProperties(image,['system:time_start'])
    collection=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(area).filterDate(args.start.isoformat(),args.end.isoformat()).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',80)).map(indices)
    start,end=ee.Date(args.start.isoformat()),ee.Date(args.end.isoformat())
    count=end.difference(start,'day').divide(7).ceil()
    empty=ee.Image.constant([0,0,0]).rename(BANDS).updateMask(ee.Image(0))
    def weekly(offset):
        begin=start.advance(ee.Number(offset).multiply(7),'day')
        finish=ee.Date(begin.advance(7,'day').millis().min(end.millis()))
        subset=collection.filterDate(begin,finish)
        composite=ee.Image(ee.Algorithms.If(subset.size().gt(0),subset.median(),empty))
        valid=composite.select('ndwi').mask().unmask(0, sameFootprint=False).rename('valid_pixel_fraction')
        reduced=composite.addBands(valid).reduceRegions(collection=grid,reducer=ee.Reducer.mean(),scale=args.scale,crs=args.crs,tileScale=4)
        def decorate(feature):
            feature=ee.Feature(feature)
            # Equality to null rather than truthiness preserves a genuine zero index.
            observed=ee.Algorithms.IsEqual(feature.get('ndwi'),None)
            return feature.set(dict(date=finish.advance(-1,'day').format('YYYY-MM-dd'),observation_start=begin.format('YYYY-MM-dd'),observation_end_exclusive=finish.format('YYYY-MM-dd'),is_observed=ee.Algorithms.If(observed,False,True),sensor='Sentinel-2_SR_HARMONIZED',source_kind='satellite_export',algorithm_version='s2-ndwi-rg-ndci-v1',source_scene_count=subset.size(),grid_m=args.grid_m,water_ndwi_min=args.water_ndwi_min))
        return reduced.map(decorate)
    table=ee.FeatureCollection(ee.List.sequence(0,count.subtract(1)).map(weekly)).flatten()
    return table

def main():
    args=arguments()
    if args.end<=args.start:
        raise SystemExit('--end must be after --start.')
    if args.grid_m<=0 or args.scale<=0:
        raise SystemExit('Grid and reduction sizes must be positive meters.')
    if args.crs=='EPSG:4326':
        raise SystemExit('Use a local projected metric CRS for the grid.')
    try:
        import ee
    except ImportError as exc:
        raise SystemExit('Install optional dependency: pip install earthengine-api') from exc
    if args.authenticate:
        ee.Authenticate()
    ee.Initialize(project=args.project)
    table=build_export(ee,args)
    selectors=['date','lake_id','zone_id','zone_name','cell_id','lat','lon']+BANDS+['valid_pixel_fraction','is_observed','sensor','source_kind','algorithm_version','source_scene_count','grid_m','water_ndwi_min','observation_start','observation_end_exclusive']
    task=ee.batch.Export.table.toDrive(collection=table,description='AquaLens_weekly_S2',folder=args.folder,fileNamePrefix=f'{args.lake_id}_S2_weekly',fileFormat='CSV',selectors=selectors)
    task.start()
    print(json.dumps(dict(task_id=task.id,state=task.status().get('state'),note='Export queued in your Earth Engine account. Download the CSV from Drive when complete. Proxy indices require local validation; weather not included.'),indent=2))

if __name__=='__main__':
    main()

