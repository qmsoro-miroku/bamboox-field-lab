/* Public demo adapter. Real ONNX inference and private per-browser IndexedDB.
 * The original FastAPI endpoints remain available in the local Python app. */
(() => {
  'use strict';
  const base = window.BAMBOOX_BASE;
  const nativeFetch = window.fetch.bind(window);
  const json = (data, status = 200) => new Response(JSON.stringify(data), {status,headers:{'Content-Type':'application/json'}});
  const fail = (message, status=400) => Object.assign(new Error(message), {status});
  const round = (n, d=2) => Number(n.toFixed(d));
  const meta = nativeFetch(base+'/models/metadata.json').then(r=>{if(!r.ok)throw fail('Model information is unavailable.',503);return r.json();});
  let datasetPromise, dbPromise, speciesSession, growthSession, inferenceQueue=Promise.resolve();
  const urls = new Map();
  const dataset = () => datasetPromise ||= nativeFetch(base+'/data/dataset.json').then(r=>r.json());
  function db() {
    return dbPromise ||= new Promise((resolve,reject)=>{
      const req=indexedDB.open('bamboox-web-journal-v1',1);
      req.onupgradeneeded=()=>req.result.createObjectStore('records',{keyPath:'id',autoIncrement:true});
      req.onsuccess=()=>resolve(req.result);
      req.onerror=()=>reject(fail('Browser storage is unavailable. Allow site storage to save your journal.',507));
    });
  }
  async function transaction(mode, fn) {
    const connection=await db();
    return new Promise((resolve,reject)=>{
      const tx=connection.transaction('records',mode), store=tx.objectStore('records');
      const req=fn(store);let result;
      if(req)req.onsuccess=()=>{result=req.result;};
      tx.oncomplete=()=>resolve(result);
      tx.onerror=()=>reject(fail('Your browser could not save this record. Storage may be full.',507));
      tx.onabort=()=>reject(fail('The journal change was not saved.',507));
    });
  }
  const records=()=>transaction('readonly',s=>s.getAll());
  function recordURL(record) {
    if(!urls.has(record.id))urls.set(record.id,URL.createObjectURL(record.imageBlob));
    return urls.get(record.id);
  }
  function serialize(record) {
    if(!record)return null;
    const {imageBlob,...data}=record;
    return {...data,image_saved:!!imageBlob,image_url:imageBlob?recordURL(record):null};
  }
  async function image(file) {
    if(!(file instanceof Blob)||!file.size)throw fail('Select a non-empty bamboo image.');
    if(file.size>10*1024*1024)throw fail('Image exceeds the 10 MB limit.',413);
    const bytes=new Uint8Array(await file.slice(0,16).arrayBuffer());
    const format=bytes[0]===255&&bytes[1]===216?'JPEG':bytes[0]===137&&bytes[1]===80&&bytes[2]===78&&bytes[3]===71?'PNG':String.fromCharCode(...bytes.slice(0,4))==='RIFF'&&String.fromCharCode(...bytes.slice(8,12))==='WEBP'?'WEBP':null;
    if(!format)throw fail('Only JPG, PNG, and WEBP images are supported.');
    let bitmap;
    try{bitmap=await createImageBitmap(file);}catch{throw fail('Invalid or corrupted image.');}
    if(bitmap.width*bitmap.height>25000000){bitmap.close();throw fail('Image dimensions exceed 25 megapixels.');}
    return {bitmap,format,width:bitmap.width,height:bitmap.height};
  }
  function tensorFor(bitmap,size,letterbox) {
    const canvas=document.createElement('canvas');canvas.width=canvas.height=size;
    const ctx=canvas.getContext('2d',{willReadFrequently:true});
    ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality=letterbox?'low':'high';
    let ratio=1, padX=0,padY=0;
    if(letterbox){
      ratio=Math.min(size/bitmap.width,size/bitmap.height);
      const w=Math.round(bitmap.width*ratio),h=Math.round(bitmap.height*ratio);
      padX=Math.round((size-w)/2-.1);padY=Math.round((size-h)/2-.1);
      ctx.fillStyle='rgb(114,114,114)';ctx.fillRect(0,0,size,size);ctx.drawImage(bitmap,padX,padY,w,h);
    }else ctx.drawImage(bitmap,0,0,size,size);
    const rgba=ctx.getImageData(0,0,size,size).data, area=size*size, data=new Float32Array(3*area);
    const means=[.485,.456,.406],std=[.229,.224,.225];
    for(let p=0;p<area;p++)for(let c=0;c<3;c++)data[c*area+p]=letterbox?rgba[p*4+c]/255:(rgba[p*4+c]/255-means[c])/std[c];
    return {tensor:new ort.Tensor('float32',data,[1,3,size,size]),ratio,padX,padY};
  }
  async function session(kind) {
    ort.env.wasm.wasmPaths=base+'/assets/ort/';ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;
    const status=kind==='species'?'Loading species model for this browser…':'Loading growth model for this browser…';
    const message=document.querySelector('#browserAIStatus');if(message)message.textContent=status;
    try {
      return await ort.InferenceSession.create(base+`/models/${kind}.onnx`,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
    }catch(error){console.error('Browser ONNX initialization failed:',error);throw fail('The AI model could not load. Check your connection and use a current Chrome, Edge, Firefox, or Safari browser.',503);}
  }
  function serialInference(fn){const job=inferenceQueue.then(fn,fn);inferenceQueue=job.catch(()=>{});return job;}
  async function species(file) {
    return serialInference(async()=>{
      const decoded=await image(file);
      try {
        speciesSession ||= session('species').catch(e=>{speciesSession=null;throw e;});
        const engine=await speciesSession,m=await meta,input=tensorFor(decoded.bitmap,224,false),start=performance.now();
        const result=await engine.run({[engine.inputNames[0]]:input.tensor});
        const logits=result[engine.outputNames[0]].data,max=Math.max(...logits),exp=Array.from(logits,x=>Math.exp(x-max)),sum=exp.reduce((a,b)=>a+b,0);
        const order=exp.map((value,index)=>({index,score:value/sum})).sort((a,b)=>b.score-a.score);
        return {predicted_species:m.species_classes[order[0].index],confidence_percent:round(order[0].score*100),
          inference_time_ms:round(performance.now()-start),top_predictions:order.slice(0,3).map(x=>({species:m.species_classes[x.index],confidence_percent:round(x.score*100)})),model:'MobileNetV3-Small · Browser ONNX',classification_type:'species'};
      }finally{decoded.bitmap.close();}
    });
  }
  function iou(a,b){const w=Math.max(0,Math.min(a.x2,b.x2)-Math.max(a.x1,b.x1)),h=Math.max(0,Math.min(a.y2,b.y2)-Math.max(a.y1,b.y1));const overlap=w*h;return overlap/Math.max(1e-8,(a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-overlap);}
  async function growth(file) {
    return serialInference(async()=>{
      const decoded=await image(file);
      try {
        growthSession ||= session('growth').catch(e=>{growthSession=null;throw e;});
        const engine=await growthSession,m=await meta,input=tensorFor(decoded.bitmap,640,true),start=performance.now();
        const result=await engine.run({[engine.inputNames[0]]:input.tensor}),raw=result[engine.outputNames[0]],data=raw.data;
        if(raw.dims.length!==3||raw.dims[1]!==7)throw fail('Unexpected growth-model output. Please reload the site.',503);
        const anchors=raw.dims[2],candidates=[];
        for(let i=0;i<anchors;i++){
          let score=0,cls=-1;
          for(let c=0;c<3;c++){const value=data[(4+c)*anchors+i];if(value>score){score=value;cls=c;}}
          if(score<m.growth_confidence)continue;
          const cx=data[i],cy=data[anchors+i],w=data[2*anchors+i],h=data[3*anchors+i];
          candidates.push({class_id:cls,confidence:score,x1:cx-w/2,y1:cy-h/2,x2:cx+w/2,y2:cy+h/2});
        }
        candidates.sort((a,b)=>b.confidence-a.confidence);
        const kept=[];
        for(const box of candidates.slice(0,30000)){
          if(kept.some(other=>other.class_id===box.class_id&&iou(other,box)>m.growth_iou))continue;
          kept.push(box);if(kept.length>=300)break;
        }
        const counts={matured:0,old:0,young:0};
        const detections=kept.map(box=>{
          const name=m.growth_classes[box.class_id];counts[name]++;
          return {class_id:box.class_id,class_name:name,confidence:round(box.confidence,6),confidence_percent:round(box.confidence*100),bbox:{
            x1:round(Math.max(0,(box.x1-input.padX)/input.ratio)),y1:round(Math.max(0,(box.y1-input.padY)/input.ratio)),
            x2:round(Math.min(decoded.width,(box.x2-input.padX)/input.ratio)),y2:round(Math.min(decoded.height,(box.y2-input.padY)/input.ratio))}};
        });
        return {status:'success',task:'growth_stage_detection',model:'YOLOv8n · Browser ONNX',image_width:decoded.width,image_height:decoded.height,
          confidence_threshold:m.growth_confidence,detection_count:detections.length,class_counts:counts,detections,inference_time_ms:round(performance.now()-start)};
      }finally{decoded.bitmap.close();}
    });
  }
  function filterRecords(all,query) {
    const q=(query.get('q')||'').trim().toLowerCase(),filter=query.get('species_filter');
    const matches=all.filter(r=>(!q||[r.sample_id,r.predicted_species,r.origin].some(x=>(x||'').toLowerCase().includes(q)))&&(!filter||r.predicted_species===filter));
    const sort=query.get('sort')||'date_desc';
    matches.sort((a,b)=>sort==='date_asc'?a.id-b.id:sort==='confidence_desc'?b.confidence_score-a.confidence_score:sort==='confidence_asc'?a.confidence_score-b.confidence_score:b.id-a.id);
    return matches;
  }
  const csvCell=value=>{let v=String(value??'');if(/^[\s]*[=+@-]/.test(v))v="'"+v;return '"'+v.replaceAll('"','""')+'"';};
  const csv=rows=>new Response('\ufeff'+rows.map(row=>row.map(csvCell).join(',')).join('\r\n'),{headers:{'Content-Type':'text/csv;charset=utf-8'}});
  async function api(path,query,method,form) {
    if(path==='/api/upload-image'&&method==='POST'){
      const decoded=await image(form.get('file'));decoded.bitmap.close();
      if(!String(form.get('sample_id')||'').trim())throw fail('Sample ID is required.');
      return json({status:'validated',image_format:decoded.format,image_width:decoded.width,image_height:decoded.height,saved_to_database:false});
    }
    if(path==='/api/predict-species'&&method==='POST')return json({status:'success',...await species(form.get('file'))});
    if(path==='/api/detect-growth-stage'&&method==='POST')return json(await growth(form.get('file')));
    if(path==='/api/inspections/next-id'){
      const all=await records();let max=0;for(const r of all){const match=/^BAM-(\d+)$/i.exec(r.sample_id);if(match)max=Math.max(max,Number(match[1]));}
      return json({suggested_id:'BAM-'+String(max+1).padStart(4,'0'),reserved:false});
    }
    if(path==='/api/inspections/species')return json({species:[...new Set((await records()).map(r=>r.predicted_species))].sort()});
    if(path==='/api/inspections'&&method==='POST'){
      const sample_id=String(form.get('sample_id')||'').trim(),origin=String(form.get('origin')||''),notes=String(form.get('notes')||''),knownSpecies=String(form.get('species')||'');
      if(!sample_id||sample_id.length>80||origin.length>250||notes.length>2000||knownSpecies.length>100)throw fail('Check the Sample ID and text field lengths.');
      const file=form.get('file'),prediction=await species(file),all=await records();
      if(all.some(r=>r.sample_id.toLowerCase()===sample_id.toLowerCase()))throw fail('This Sample ID already exists in your journal.',409);
      const record={sample_id,species:knownSpecies,origin,notes,filename:file.name||'capture.png',predicted_species:prediction.predicted_species,
        confidence_score:prediction.confidence_percent,inference_time_ms:prediction.inference_time_ms,model_name:prediction.model,classification_type:'species',
        inspection_date:new Date().toISOString(),imageBlob:file};
      const id=await transaction('readwrite',store=>store.add(record));
      return json({status:'saved',inspection_id:id,sample_id,predicted_species:prediction.predicted_species,confidence_percent:prediction.confidence_percent,image_saved:true},201);
    }
    if(path==='/api/inspections'&&method==='GET'){
      const all=filterRecords(await records(),query),offset=Math.max(0,Number(query.get('offset'))||0),limit=Math.max(1,Math.min(100,Number(query.get('limit'))||50));
      return json({count:all.slice(offset,offset+limit).length,total:all.length,offset,limit,has_more:offset+limit<all.length,records:all.slice(offset,offset+limit).map(serialize)});
    }
    if(path==='/api/inspections/export.csv'){
      const fields=['id','sample_id','species','origin','notes','filename','predicted_species','confidence_score','inference_time_ms','model_name','classification_type','inspection_date'];
      return csv([fields,...filterRecords(await records(),query).map(r=>fields.map(f=>r[f]))]);
    }
    const recordMatch=/^\/api\/inspections\/(\d+)(\/image)?$/.exec(path);
    if(recordMatch){
      const id=Number(recordMatch[1]),record=await transaction('readonly',s=>s.get(id));
      if(!record)throw fail('Inspection record not found in this browser.',404);
      if(method==='DELETE'){
        await transaction('readwrite',s=>s.delete(id));if(urls.has(id)){URL.revokeObjectURL(urls.get(id));urls.delete(id);}
        return json({status:'deleted',inspection_id:id,image_removed:true});
      }
      return recordMatch[2]?new Response(record.imageBlob):json(serialize(record));
    }
    if(path==='/api/dashboard/summary'){
      const all=await records(),counts={};all.forEach(r=>counts[r.predicted_species]=(counts[r.predicted_species]||0)+1);
      const avg=key=>all.length?round(all.reduce((sum,r)=>sum+Number(r[key]||0),0)/all.length):null;
      return json({status:'success',total_inspections:all.length,species_identified:Object.keys(counts).length,average_confidence:avg('confidence_score'),
        average_inference_ms:avg('inference_time_ms'),species_distribution:Object.entries(counts).map(([species,total])=>({species,total})),
        recent_inspections:[...all].sort((a,b)=>b.id-a.id).slice(0,5).map(serialize),growth_dataset:await dataset(),classification_type:'species_demo'});
    }
    if(path.startsWith('/api/growth-dataset/')){
      const data=await dataset();
      if(path.endsWith('/summary')){const {images,...summary}=data;return json(summary);}
      if(path.endsWith('/images')){
        const split=query.get('split')||'all',stage=query.get('stage')||'all';
        return json({...data,images:data.images.filter(i=>(split==='all'||i.split===split)&&(stage==='all'||stage==='needs_review'&&['missing_label','invalid_label'].includes(i.annotation_status)||i.objects.some(b=>b.class_name===stage)))});
      }
      if(path.endsWith('/export'))return csv([['filename','split','class_id','class_name','x_center','y_center','width','height','annotation_status'],...data.images.flatMap(i=>(i.objects.length?i.objects:[null]).map(b=>[i.filename,i.split,...(b?[b.class_id,b.class_name,b.x_center,b.y_center,b.width,b.height]:['','','','','','']),i.annotation_status]))]);
    }
    if(path==='/api/settings/overview'){
      const all=await records(),data=await dataset(),m=await meta,bytes=all.reduce((sum,r)=>sum+(r.imageBlob?.size||0),0);
      return json({application:'BambooX',version:'3.0 public browser demo',checked_at_utc:new Date().toISOString(),backend:{status:'running',execution:'Browser WebAssembly'},
        model:{checkpoint_present:true,checkpoint_size_bytes:m.species_model_bytes},growth_model:{checkpoint_present:true,checkpoint_size_bytes:m.growth_model_bytes},
        database:{exists:true,status:'ready',record_count:all.length,size_bytes:bytes,path:'This browser · IndexedDB · bamboox-web-journal-v1'},
        storage:{uploads:{count:all.length,bytes},growth_dataset:data}});
    }
    throw fail('This operation is not available in the public browser demo.',404);
  }
  window.fetch=async function(input,options={}){
    const url=new URL(typeof input==='string'?input:input.url,location.href);
    let path=url.pathname;if(base&&path.startsWith(base+'/'))path=path.slice(base.length);
    if(url.origin!==location.origin||!path.startsWith('/api/'))return nativeFetch(input,options);
    if(options.signal?.aborted)throw new DOMException('Aborted','AbortError');
    try{
      const response=await api(path,url.searchParams,(options.method||'GET').toUpperCase(),options.body);
      if(options.signal?.aborted)throw new DOMException('Aborted','AbortError');
      return response;
    }catch(error){if(error.name==='AbortError')throw error;return json({detail:error.message||'This browser could not complete the request.'},error.status||500);}
  };
  window.bambooxDownload=async function(url,filename){
    const response=await fetch(url);if(!response.ok){window.bambooxToast?.('Export failed.','error');return;}
    const blob=await response.blob(),href=URL.createObjectURL(blob),a=document.createElement('a');a.href=href;a.download=filename;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(href),30000);
  };
  document.addEventListener('click',event=>{
    const a=event.target.closest('a');if(!a)return;
    const url=new URL(a.href,location.href);
    if(url.pathname.endsWith('/api/growth-dataset/export')){event.preventDefault();window.bambooxDownload(a.href,'bamboox_growth_annotations.csv');}
  });
  window.addEventListener('pagehide',()=>{for(const url of urls.values())URL.revokeObjectURL(url);});
})();
