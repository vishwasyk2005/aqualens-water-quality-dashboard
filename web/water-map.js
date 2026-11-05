/* AquaLens geographic observation explorer. Leaflet is vendored locally. */
(function(global){
  'use strict';
  const PARAMETERS=['ndwi','turbidity_proxy','chlorophyll_proxy'];
  const NAMES={ndwi:'NDWI',turbidity_proxy:'Turbidity proxy',chlorophyll_proxy:'Chlorophyll proxy'};
  const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number=(v,d=3)=>v==null||!Number.isFinite(+v)?'—':Number(v).toLocaleString('en-US',{maximumFractionDigits:d,minimumFractionDigits:d});
  const day=value=>String(value??'').slice(0,10);
  const humanDate=value=>value?new Date(day(value)+'T12:00:00').toLocaleDateString('en-GB',{day:'numeric',month:'short',year:'numeric'}):'—';
  function defaultValid(row,parameter){
    const value=row[parameter];
    return value!==null&&value!==undefined&&value!==''&&Number.isFinite(+value)&&
      (parameter!=='ndwi'||(+value>=-1&&+value<=1))&&
      !['false','0','no'].includes(String(row[parameter+'_is_observed']??row.is_observed).toLowerCase())&&
      (row.valid_pixel_fraction==null||+row.valid_pixel_fraction>=.5);
  }
  function selectReadings(rows,{date,parameter,scope='all',validRow=defaultValid}){
    return rows.filter(row=>day(row.date)===date&&(scope==='all'||String(row.zone_id)===scope)&&
      row.lat!=null&&row.lon!=null&&row.lat!==''&&row.lon!==''&&Number.isFinite(+row.lat)&&Number.isFinite(+row.lon)&&
      Math.abs(+row.lat)<=90&&Math.abs(+row.lon)<=180&&validRow(row,parameter));
  }
  function colorDomain(rows,parameter){
    const values=rows.map(row=>+row[parameter]).filter(Number.isFinite).sort((a,b)=>a-b);
    return values.length?{min:values[0],max:values.at(-1)}:{min:0,max:1};
  }
  function color(value,domain){
    const t=Math.max(0,Math.min(1,(value-domain.min)/(domain.max-domain.min||1)));
    return `hsl(${184-155*t},${65+12*t}%,${37+13*t}%)`;
  }
  let active=null;
  const preference={dataset:null,date:'',scope:'all',selected:'all',base:'satellite',visible:true,opacity:.85,center:null,zoom:null};
  function markup({large=false}={}){
    return `<section class="geo-map-shell ${large?'is-large':''}" data-geo-map-shell aria-label="Interactive geographic water observation map">
      <div class="geo-map-toolbar"><div><strong>Explore the lake</strong><p>Drag to move · Scroll or pinch to zoom · Click a point for its reading</p></div><div class="toolbar"><button class="btn" type="button" data-map-fit>Fit observations</button><button class="btn" type="button" data-map-expand aria-pressed="false">Expand map</button></div></div>
      <div class="geo-map-filters"><label>Observation date <select class="select" data-map-date aria-label="Map observation date"></select></label><label>Water parameter <select class="select" data-map-parameter aria-label="Map water parameter">${PARAMETERS.map(p=>`<option value="${p}">${NAMES[p]}</option>`).join('')}</select></label><label>Zone <select class="select" data-map-zone aria-label="Map zone"><option value="all">All zones</option></select></label><label>Background <select class="select" data-map-base aria-label="Map background"><option value="satellite">Satellite</option><option value="streets">Streets</option><option value="plain">Plain</option></select></label></div>
      <div class="geo-map-filters map-time-controls"><button class="btn icon-btn" type="button" data-map-step="-1" aria-label="Previous observation">‹</button><button class="btn" type="button" data-map-play aria-label="Play observation timeline">Play dates</button><input class="map-date-range" type="range" min="0" max="0" value="0" step="1" data-map-timeline aria-label="Observation timeline"><button class="btn icon-btn" type="button" data-map-step="1" aria-label="Next observation">›</button><span class="map-stat" data-map-date-label></span></div>
      <div class="geo-map-stage"><div class="geo-map" data-geo-map role="region" aria-label="Geographic map. Use arrow keys to pan and plus or minus to zoom."></div><aside class="geo-map-inspector" aria-label="Selected water observation"><div data-map-detail><span class="eyebrow">POINT INSPECTOR</span><h3>Choose a colored point</h3><p>See the recorded values, observation date, and location.</p></div><details class="map-reading-browser"><summary>Browse readings</summary><label class="visually-hidden" for="map-reading-search">Search map readings</label><input class="input" id="map-reading-search" data-map-search placeholder="Cell ID, zone or coordinates" aria-label="Search map readings"><div data-map-reading-list></div></details></aside></div>
      <div class="geo-map-tile-warning" data-map-warning role="status" hidden>Background tiles could not load. Points and map controls still work. Try Streets or Plain.</div>
      <div class="geo-map-status" data-map-status aria-live="polite"></div><div class="geo-map-legend" data-map-legend></div>
      <div class="geo-map-filters"><label class="map-point-toggle"><input type="checkbox" data-map-visible checked> Show observations</label><label>Point opacity <input type="range" min="20" max="100" value="85" step="5" data-map-opacity aria-label="Observation point opacity"></label><span class="muted" data-map-cursor>Move across the map to see coordinates</span></div>
      <p class="geo-map-help">Points are actual uploaded observations for the selected date. Background imagery has its own acquisition dates and is not the source of these measurements. Colors show the supplied parameter scale, not a water-safety classification.</p>
    </section>`;
  }
  function destroy(){
    if(!active)return;
    if(active.timer)clearInterval(active.timer);
    if(active.resizeObserver)active.resizeObserver.disconnect();
    document.removeEventListener('keydown',active.escapeHandler);
    active.map.remove();
    document.body.classList.remove('map-expanded');
    active=null;
  }
  function mount(options){
    const root=document.querySelector('[data-geo-map-shell]');
    if(!root)return;
    if(!global.L){root.innerHTML='<div class="empty-state">The map library did not load. Refresh the dashboard to retry.</div>';return;}
    destroy();
    const rows=options.observations||[],dates=[...new Set(rows.map(row=>day(row.date)).filter(Boolean))].sort();
    if(!dates.length){root.innerHTML='<div class="empty-state">Upload dated observations to explore the map.</div>';return;}
    const selected=options.selected||'all';
    if(preference.dataset!==options.dataset){Object.assign(preference,{dataset:options.dataset,date:dates.at(-1),scope:selected,selected,center:null,zoom:null});}
    if(!dates.includes(preference.date))preference.date=dates.at(-1);
    if(preference.selected!==selected){preference.scope=selected;preference.selected=selected;preference.center=null;preference.zoom=null;}
    const query=selector=>root.querySelector(selector), mapNode=query('[data-geo-map]'),L=global.L;
    const map=L.map(mapNode,{preferCanvas:true,zoomControl:true,scrollWheelZoom:true,keyboard:true,dragging:true,touchZoom:true,doubleClickZoom:true,minZoom:2,maxZoom:19});
    const points=L.layerGroup().addTo(map);
    const tiles={
      satellite:L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxNativeZoom:19,maxZoom:19,attribution:'Imagery © Esri, Vantor, Earthstar Geographics, and the GIS User Community',referrerPolicy:'strict-origin-when-cross-origin'}),
      streets:L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxNativeZoom:19,maxZoom:19,attribution:'© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors',referrerPolicy:'strict-origin-when-cross-origin'})
    };
    map.attributionControl.setPrefix('<a href="https://leafletjs.com" target="_blank" rel="noopener">Leaflet</a>');
    L.control.scale({imperial:false,position:'bottomleft'}).addTo(map);
    active={map,root,points,rows:[],markers:[],date:preference.date,parameter:options.parameter,base:preference.base,timer:null};
    const current=active;
    const emptyDetail=()=>{query('[data-map-detail]').innerHTML='<span class="eyebrow">POINT INSPECTOR</span><h3>Choose a colored point</h3><p>See the recorded values, observation date, and location.</p>';};
    function inspect(row,marker){
      const detail=`<span class="eyebrow">OBSERVED · ${escape(humanDate(row.date))}</span><h3>${escape(row.zone_name||row.zone_id)}</h3><p class="map-cell-id">${escape(row.cell_id)}</p><dl class="map-reading-values">${PARAMETERS.map(p=>`<div><dt>${NAMES[p]}</dt><dd>${options.validRow(row,p)?number(row[p]):'Unavailable'}</dd></div>`).join('')}</dl><p>${number(row.lat,5)}° N, ${number(row.lon,5)}° E</p><p class="fine-print">${row.valid_pixel_fraction==null?'Pixel QA is not supplied; measurement quality is unverified.':`Supplied valid-pixel fraction: ${number(+row.valid_pixel_fraction*100,1)}%.`} Proxy units and retrieval algorithms need verification.</p><button class="btn btn-primary full-width" type="button" data-map-inspect-zone>Open zone analysis →</button>`;
      query('[data-map-detail]').innerHTML=detail;
      query('[data-map-inspect-zone]').onclick=()=>options.onInspectZone?.(String(row.zone_id));
      marker.openPopup();
    }
    function browse(){
      const search=query('[data-map-search]').value.trim().toLowerCase();
      const matches=current.rows.map((row,index)=>({row,index})).filter(({row})=>!search||[row.cell_id,row.zone_id,row.zone_name,number(row.lat,5),number(row.lon,5)].some(v=>String(v).toLowerCase().includes(search)));
      query('[data-map-reading-list]').innerHTML=matches.slice(0,8).map(({row,index})=>`<button type="button" class="map-reading-button" data-map-reading="${index}"><strong>${escape(row.zone_name||row.zone_id)}</strong><span>${number(row[current.parameter])} · ${escape(String(row.cell_id).slice(-12))}</span></button>`).join('')||'<p class="muted">No matching observations.</p>';
      query('[data-map-reading-list]').querySelectorAll('[data-map-reading]').forEach(button=>button.onclick=()=>{const index=+button.dataset.mapReading,row=current.rows[index],marker=current.markers[index];map.panTo([+row.lat,+row.lon]);inspect(row,marker);});
    }
    function fit(){
      const readings=current.rows.length?current.rows:selectReadings(rows,{date:preference.date,parameter:current.parameter,validRow:options.validRow});
      if(readings.length)map.fitBounds(L.latLngBounds(readings.map(row=>[+row.lat,+row.lon])),{padding:[35,35],maxZoom:15,animate:false});
      else map.setView([+rows[0].lat||0,+rows[0].lon||0],10);
    }
    function draw(){
      map.closePopup();points.clearLayers();emptyDetail();
      current.rows=selectReadings(rows,{date:preference.date,parameter:current.parameter,scope:preference.scope,validRow:options.validRow});
      current.date=preference.date;current.markers=[];
      const domain=colorDomain(current.rows,current.parameter);
      current.rows.forEach(row=>{
        const marker=L.circleMarker([+row.lat,+row.lon],{radius:5.5,color:'#fff',weight:1,fillColor:color(+row[current.parameter],domain),fillOpacity:preference.opacity,opacity:.8});
        marker.bindTooltip(`${escape(row.zone_name||row.zone_id)} · ${escape(NAMES[current.parameter])}: ${number(row[current.parameter])}`,{direction:'top'});
        marker.bindPopup(`<div class="map-popup"><strong>${escape(row.zone_name||row.zone_id)}</strong><span>${escape(NAMES[current.parameter])}</span><b>${number(row[current.parameter])}</b><small>Observed ${escape(humanDate(row.date))}</small><small>${number(row.lat,5)}°, ${number(row.lon,5)}°</small><button type="button" class="btn btn-primary" data-popup-inspect>Open zone analysis →</button></div>`,{minWidth:190,maxWidth:250});
        marker.on('click',()=>inspect(row,marker));
        marker.on('popupopen',event=>{event.popup.getElement()?.querySelector('[data-popup-inspect]')?.addEventListener('click',()=>options.onInspectZone?.(String(row.zone_id)));});
        marker.addTo(points);current.markers.push(marker);
      });
      if(preference.visible){if(!map.hasLayer(points))points.addTo(map);}else if(map.hasLayer(points))map.removeLayer(points);
      query('[data-map-status]').textContent=`${current.rows.length.toLocaleString()} valid observed points · ${humanDate(preference.date)} · ${NAMES[current.parameter]}${preference.scope==='all'?' · All zones':' · '+preference.scope}`;
      query('[data-map-legend]').innerHTML=current.rows.length?`<strong>${escape(NAMES[current.parameter])}</strong><span>${number(domain.min)}</span><i class="map-color-ramp" style="background:linear-gradient(90deg,${color(domain.min,domain)},${color(domain.max,domain)})"></i><span>${number(domain.max)}</span><span class="muted">Selected-date range · ${current.parameter==='ndwi'?'Dimensionless index':'Supplied proxy units unverified'}</span>`:'<span>No valid measurements for this date, zone and parameter.</span>';
      query('[data-map-date-label]').textContent=humanDate(preference.date);
      query('[data-map-date]').value=preference.date;
      query('[data-map-timeline]').value=dates.indexOf(preference.date);
      root.querySelectorAll('[data-map-step]').forEach(button=>button.disabled=(+button.dataset.mapStep<0&&preference.date===dates[0])||(+button.dataset.mapStep>0&&preference.date===dates.at(-1)));
      browse();
    }
    function setDate(index){preference.date=dates[Math.max(0,Math.min(dates.length-1,index))];draw();}
    function pause(){if(current.timer)clearInterval(current.timer);current.timer=null;query('[data-map-play]').textContent='Play dates';query('[data-map-play]').setAttribute('aria-label','Play observation timeline');}
    function expand(force){
      const expanded=typeof force==='boolean'?force:!root.classList.contains('geo-map-expanded');
      if(expanded)current.returnFocus=document.activeElement;
      root.classList.toggle('geo-map-expanded',expanded);document.body.classList.toggle('map-expanded',expanded);
      root.setAttribute('role',expanded?'dialog':'region');
      if(expanded){root.setAttribute('aria-modal','true');query('[data-map-expand]').focus();}else{root.removeAttribute('aria-modal');if(current.returnFocus?.isConnected)current.returnFocus.focus();}
      query('[data-map-expand]').textContent=expanded?'Close expanded map':'Expand map';query('[data-map-expand]').setAttribute('aria-pressed',String(expanded));
      requestAnimationFrame(()=>map.invalidateSize({pan:false}));
    }
    function setBase(name){
      Object.values(tiles).forEach(layer=>{if(map.hasLayer(layer))map.removeLayer(layer);});
      preference.base=name;current.base=name;query('[data-map-warning]').hidden=true;
      if(tiles[name])tiles[name].addTo(map);
      query('[data-map-base]').value=name;
    }
    Object.entries(tiles).forEach(([name,layer])=>{
      layer.on('tileerror',()=>{if(active===current&&current.base===name)query('[data-map-warning]').hidden=false;});
      layer.on('tileload',()=>{if(active===current&&current.base===name)query('[data-map-warning]').hidden=true;});
    });
    query('[data-map-date]').innerHTML=dates.map(value=>`<option value="${value}">${humanDate(value)}</option>`).join('');
    const zones=[...new Map(rows.map(row=>[String(row.zone_id),row.zone_name||row.zone_id])).entries()].sort(([a],[b])=>a.localeCompare(b));
    query('[data-map-zone]').innerHTML='<option value="all">All zones</option>'+zones.map(([id,name])=>`<option value="${escape(id)}">${escape(name)}</option>`).join('');
    if(!zones.some(([id])=>id===preference.scope))preference.scope='all';
    query('[data-map-zone]').value=preference.scope;
    query('[data-map-parameter]').value=current.parameter;
    query('[data-map-visible]').checked=preference.visible;
    query('[data-map-opacity]').value=Math.round(preference.opacity*100);
    query('[data-map-timeline]').max=dates.length-1;
    query('[data-map-date]').onchange=event=>{pause();setDate(dates.indexOf(event.target.value));};
    query('[data-map-timeline]').oninput=event=>{pause();setDate(+event.target.value);};
    root.querySelectorAll('[data-map-step]').forEach(button=>button.onclick=()=>{pause();setDate(dates.indexOf(preference.date)+ +button.dataset.mapStep);});
    query('[data-map-play]').onclick=()=>{
      if(current.timer){pause();return;}
      if(preference.date===dates.at(-1))setDate(0);
      query('[data-map-play]').textContent='Pause';query('[data-map-play]').setAttribute('aria-label','Pause observation timeline');
      current.timer=setInterval(()=>{const index=dates.indexOf(preference.date)+1;if(index>=dates.length){pause();return;}setDate(index);},1100);
    };
    query('[data-map-zone]').onchange=event=>{preference.scope=event.target.value;draw();fit();};
    query('[data-map-parameter]').onchange=event=>{pause();options.onParameterChange?.(event.target.value);};
    query('[data-map-base]').onchange=event=>setBase(event.target.value);
    query('[data-map-visible]').onchange=event=>{preference.visible=event.target.checked;if(preference.visible)points.addTo(map);else map.removeLayer(points);};
    query('[data-map-opacity]').oninput=event=>{preference.opacity=+event.target.value/100;current.markers.forEach(marker=>marker.setStyle({fillOpacity:preference.opacity}));};
    query('[data-map-fit]').onclick=fit;
    query('[data-map-expand]').onclick=()=>expand();
    query('[data-map-search]').oninput=browse;
    current.escapeHandler=event=>{
      if(!root.classList.contains('geo-map-expanded'))return;
      if(event.key==='Escape'){event.preventDefault();expand(false);return;}
      if(event.key==='Tab'){
        const focusable=[...root.querySelectorAll('button:not([disabled]),select:not([disabled]),input:not([disabled]),a[href],summary,[tabindex="0"]')].filter(node=>node.offsetParent!==null);
        const first=focusable[0],last=focusable.at(-1),focused=document.activeElement;
        if(!root.contains(focused)||(!event.shiftKey&&focused===last)){event.preventDefault();first?.focus();}
        else if(event.shiftKey&&focused===first){event.preventDefault();last?.focus();}
      }
    };
    document.addEventListener('keydown',current.escapeHandler);
    map.on('moveend',()=>{const center=map.getCenter();preference.center=[center.lat,center.lng];preference.zoom=map.getZoom();mapNode.setAttribute('data-center',`${center.lat.toFixed(6)},${center.lng.toFixed(6)}`);mapNode.setAttribute('data-zoom',String(map.getZoom()));});
    map.on('mousemove',event=>{query('[data-map-cursor]').textContent=`${number(event.latlng.lat,5)}°, ${number(event.latlng.lng,5)}°`;});
    draw();
    if(preference.center&&preference.zoom!=null)map.setView(preference.center,preference.zoom,{animate:false});else fit();
    setBase(preference.base);
    if(global.ResizeObserver){current.resizeObserver=new ResizeObserver(()=>{if(active===current)map.invalidateSize({pan:false});});current.resizeObserver.observe(mapNode);}
    requestAnimationFrame(()=>{if(active===current)map.invalidateSize({pan:false});});
  }
  global.AquaLensMap={markup,mount,destroy,selectReadings,colorDomain};
})(window);
