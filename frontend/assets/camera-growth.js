/* Same growth API for FastAPI and the public browser adapter. One request at a time. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const video=$('cameraVideo'), photo=$('capturePreview'), stage=$('cameraStage');
  const start=$('startLiveGrowth'), stop=$('stopLiveGrowth'), still=$('detectCapturedGrowth');
  const status=$('growthCameraStatus'), report=$('growthCameraReport');
  if(!video||!photo||!stage||!start||!stop||!still) return;
  const NS='http://www.w3.org/2000/svg', colors={matured:'#7ee4a0',old:'#ffbd70',young:'#85c8ff'};
  const overlay=document.createElementNS(NS,'svg');
  overlay.id='growthCameraOverlay';overlay.setAttribute('aria-hidden','true');
  overlay.setAttribute('preserveAspectRatio','xMidYMid meet');stage.append(overlay);
  const canvas=document.createElement('canvas');
  let active=false, pending=false, generation=0, timer, busy=false, recording=false, scans=0;
  const hasPhoto=()=>!photo.hidden&&!!photo.getAttribute('src')&&photo.naturalWidth>0;
  const cameraOn=()=>!!video.srcObject?.getVideoTracks().some(t=>t.readyState==='live');
  const liveReady=()=>cameraOn()&&!video.hidden&&video.readyState>=2&&video.videoWidth>0;
  function message(text,kind=''){status.textContent=text;status.dataset.kind=kind;}
  function clearOverlay(){overlay.replaceChildren();overlay.style.display='none';delete overlay.dataset.mode;}
  function controls(){
    start.disabled=!cameraOn()||busy||recording||active||pending||document.hidden;
    stop.disabled=!active;
    still.disabled=!hasPhoto()||active||pending||busy||recording;
    start.textContent=active?'Live YOLO running':'Start Live YOLO';
  }
  function stopLive(text){
    active=false;generation++;clearTimeout(timer);clearOverlay();
    if(text)message(text);
    controls();
  }
  function element(tag,attributes){
    const node=document.createElementNS(NS,tag);
    for(const [key,value] of Object.entries(attributes))node.setAttribute(key,String(value));
    return node;
  }
  function draw(data){
    overlay.replaceChildren();overlay.setAttribute('viewBox',`0 0 ${data.image_width} ${data.image_height}`);
    // Scale label typography with source pixels so it remains readable on portrait frames.
    const unit=Math.max(data.image_width,data.image_height)/1000;
    for(const detection of data.detections){
      const b=detection.bbox,color=colors[detection.class_name]||'#fff';
      overlay.append(element('rect',{x:b.x1,y:b.y1,width:Math.max(0,b.x2-b.x1),height:Math.max(0,b.y2-b.y1),fill:'none',stroke:color,'stroke-width':3*unit}));
      const label=`${detection.class_name.toUpperCase()} ${Number(detection.confidence_percent).toFixed(1)}%`;
      const width=Math.min(data.image_width,label.length*12*unit+16*unit),height=30*unit;
      const x=Math.max(0,Math.min(b.x1,data.image_width-width)),y=Math.max(0,Math.min(b.y1-height,data.image_height-height));
      overlay.append(element('rect',{x,y,width,height,rx:3*unit,fill:color}));
      const text=element('text',{x:x+7*unit,y:y+22*unit,fill:'#10251f','font-size':20*unit,'font-weight':700});
      text.textContent=label;overlay.append(text);
    }
    overlay.style.display='block';
  }
  function show(data,live,elapsed){
    draw(data);
    const count=data.detection_count,counts=data.class_counts;
    const threshold=Math.round(data.confidence_threshold*100);
    const progress=live?`Live YOLO · frame ${++scans} · ${(1000/Math.max(500,elapsed+150)).toFixed(1)} scans/s`:'Captured frame · YOLOv8n';
    message(count?`${progress} · ${count} object${count===1?'':'s'}`:`${progress} · No detections above ${threshold}% confidence.`,count?'success':'');
    report.textContent=`Young ${counts.young} · Matured ${counts.matured} · Old ${counts.old} · Model ${data.inference_time_ms} ms`;
    report.hidden=false;
    $('growthCameraHint').hidden=count>0;
    // A result may only stay on screen while its original frame/view is current.
    overlay.dataset.mode=live?'live':'capture';
  }
  async function detect(blob){
    const form=new FormData();form.append('file',blob,'bamboox_camera_frame.png');
    const response=await fetch('/api/detect-growth-stage',{method:'POST',body:form});
    const data=await response.json();
    if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:`Detection failed (${response.status}).`);
    return data;
  }
  async function nextFrame(token){
    if(!active||token!==generation)return;
    if(document.hidden||!liveReady()){stopLive('Live YOLO paused. Return to the camera and press Start Live YOLO.');return;}
    pending=true;controls();const began=performance.now();
    try{
      // Keep source dimensions. Preprocessing in the model adapter handles letterboxing.
      canvas.width=video.videoWidth;canvas.height=video.videoHeight;
      canvas.getContext('2d').drawImage(video,0,0);
      const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/png'));
      if(!blob)throw new Error('Could not read this camera frame.');
      if(!active||token!==generation)return;
      const data=await detect(blob);
      if(active&&token===generation&&liveReady())show(data,true,performance.now()-began);
    }catch(error){
      if(token===generation)stopLive('Live YOLO stopped: '+error.message);
    }finally{
      pending=false;controls();
      // Schedule after completion. Slow phones cannot accumulate a queue of old frames.
      if(active&&token===generation)timer=setTimeout(()=>nextFrame(token),Math.max(150,500-(performance.now()-began)));
    }
  }
  start.addEventListener('click',()=>{
    if(start.disabled)return;
    document.dispatchEvent(new CustomEvent('bamboox:camera-live'));
    if(!liveReady()){message('Camera is warming up. Try Start Live YOLO again.');return;}
    clearOverlay();report.hidden=true;scans=0;active=true;const token=++generation;
    message('Loading YOLO / scanning the current frame… First use downloads the model.');
    controls();nextFrame(token);
  });
  stop.addEventListener('click',()=>stopLive('Live YOLO stopped. Camera preview is still on.'));
  still.addEventListener('click',async()=>{
    if(still.disabled||!hasPhoto())return;
    const source=photo.src,token=++generation;pending=true;clearOverlay();controls();
    message('Scanning the captured frame… First use downloads the model.');
    const began=performance.now();
    try{
      const response=await fetch(source);if(!response.ok)throw new Error('Captured photo is unavailable.');
      const data=await detect(await response.blob());
      if(token===generation&&hasPhoto()&&photo.src===source)show(data,false,performance.now()-began);
    }catch(error){if(token===generation)message('Growth detection failed: '+error.message,'error');}
    finally{pending=false;controls();}
  });
  document.addEventListener('bamboox:camera-state',event=>{
    busy=event.detail.busy;recording=event.detail.recording;
    if(active&&(!event.detail.active||busy||recording))stopLive('Live YOLO paused for the camera action.');
    if(!cameraOn()&&overlay.dataset.mode==='live')clearOverlay();
    controls();
  });
  new MutationObserver(()=>{
    if(hasPhoto()){
      stopLive('Captured frame ready. Choose Detect Captured Frame for growth boxes.');
      report.hidden=true;
    }else if(overlay.dataset.mode==='capture'){
      generation++;clearOverlay();report.hidden=true;
    }
    controls();
  }).observe(photo,{attributes:true,attributeFilter:['src','hidden']});
  photo.addEventListener('load',controls);
  video.addEventListener('loadeddata',controls);
  for(const id of ['captureFrame','recordClip','stopCamera'])$(id).addEventListener('click',()=>{
    stopLive();report.hidden=true;
    if(id==='stopCamera')message('Camera stopped. Capture or restart the camera to scan.');
  });
  document.addEventListener('visibilitychange',()=>{
    if(document.hidden&&active)stopLive('Live YOLO paused while this tab is hidden. Press Start Live YOLO to resume.');
    controls();
  });
  window.addEventListener('pagehide',()=>stopLive());
  controls();
})();
