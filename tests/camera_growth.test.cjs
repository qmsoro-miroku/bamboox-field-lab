/* Controller regressions. DOM and API fixtures isolate scheduling from inference. */
const {test}=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../frontend/assets/camera-growth.js'),'utf8');
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
class Element{
  constructor(){this.listeners={};this.style={};this.dataset={};this.attributes={};this.children=[];this.hidden=false;this.disabled=false;}
  addEventListener(type,fn){(this.listeners[type]||=[]).push(fn);}
  dispatchEvent(event){for(const fn of this.listeners[event.type]||[])fn(event);}
  click(){if(!this.disabled)this.dispatchEvent({type:'click'});}
  setAttribute(key,value){this.attributes[key]=value;}
  getAttribute(key){return this.attributes[key];}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
}
function setup(){
  const ids=['cameraVideo','capturePreview','cameraStage','startLiveGrowth','stopLiveGrowth','detectCapturedGrowth','growthCameraStatus','growthCameraReport','growthCameraHint','captureFrame','recordClip','stopCamera'];
  const nodes=Object.fromEntries(ids.map(id=>[id,new Element()]));
  const document=new Element(),window=new Element(),timers=new Set();let observer;
  document.hidden=false;document.getElementById=id=>nodes[id];document.createElementNS=()=>new Element();
  document.createElement=()=>({getContext:()=>({drawImage(){}}),toBlob:cb=>cb(new Blob(['frame']))});
  const video=nodes.cameraVideo,photo=nodes.capturePreview;
  video.hidden=true;video.readyState=2;video.videoWidth=640;video.videoHeight=480;video.srcObject=null;
  photo.hidden=true;photo.naturalWidth=0;
  const result={calls:0,inflight:0,maxInflight:0,blank:false,lag:10};
  const fetch=async url=>{
    if(String(url).startsWith('blob:'))return {ok:true,blob:async()=>new Blob(['capture'])};
    result.calls++;result.maxInflight=Math.max(result.maxInflight,++result.inflight);
    const blank=result.blank;await delay(result.lag);result.inflight--;
    return {ok:true,json:async()=>({detection_count:blank?0:1,image_width:640,image_height:480,confidence_threshold:.25,inference_time_ms:10,class_counts:{young:blank?0:1,matured:0,old:0},detections:blank?[]:[{class_name:'young',confidence_percent:80,bbox:{x1:12,y1:24,x2:210,y2:440}}]})};
  };
  document.addEventListener('bamboox:camera-live',()=>{video.hidden=false;photo.hidden=true;});
  vm.runInNewContext(source,{document,window,Blob,FormData,fetch,performance,CustomEvent:class{constructor(type){this.type=type;}},
    MutationObserver:class{constructor(cb){observer=cb;}observe(){}},
    setTimeout:(fn,ms)=>{const id=setTimeout(()=>{timers.delete(id);fn();},Math.min(ms,15));timers.add(id);return id;},clearTimeout});
  return {nodes,document,result,observer:()=>observer(),connect:()=>{video.srcObject={getVideoTracks:()=>[{readyState:'live'}]};document.dispatchEvent({type:'bamboox:camera-state',detail:{active:true,busy:false,recording:false}});},cleanup:()=>{window.dispatchEvent({type:'pagehide'});for(const id of timers)clearTimeout(id);}};
}
async function until(fn){for(let i=0;i<100;i++){if(fn())return;await delay(10);}assert.fail('Condition timed out');}
test('Live frames run serially, blank frames remove boxes, and late responses cannot undo Stop',async()=>{
  const t=setup();try{
    assert.equal(t.nodes.startLiveGrowth.disabled,true);t.connect();t.nodes.startLiveGrowth.click();
    await until(()=>t.result.calls>=3&&t.nodes.growthCameraStatus.textContent.includes('frame'));
    const overlay=t.nodes.cameraStage.children[0];assert.equal(overlay.children.length,3);
    t.result.blank=true;await until(()=>t.nodes.growthCameraStatus.textContent.includes('No detections above 25%'));
    assert.equal(overlay.children.length,0);assert.equal(t.result.maxInflight,1);
    t.result.lag=120;await until(()=>t.result.inflight===1);
    t.nodes.stopLiveGrowth.click();const atStop=t.result.calls;
    await delay(150);assert.equal(t.result.calls,atStop);
    assert.match(t.nodes.growthCameraStatus.textContent,/stopped/);assert.equal(overlay.style.display,'none');
  }finally{t.cleanup();}
});
test('New captures ignore old detections; capture inference does not require a running camera',async()=>{
  const t=setup();try{
    const photo=t.nodes.capturePreview;photo.hidden=false;photo.naturalWidth=640;photo.src='blob:first';photo.attributes.src=photo.src;t.observer();
    assert.equal(t.nodes.detectCapturedGrowth.disabled,false);t.result.lag=80;t.nodes.detectCapturedGrowth.click();
    await until(()=>t.result.inflight===1);photo.src='blob:second';photo.attributes.src=photo.src;t.observer();
    await until(()=>!t.nodes.detectCapturedGrowth.disabled);assert.equal(t.nodes.cameraStage.children[0].style.display,'none');
    assert.match(t.nodes.growthCameraStatus.textContent,/Captured frame ready/);
    t.nodes.detectCapturedGrowth.click();await until(()=>t.nodes.growthCameraStatus.textContent.includes('Captured frame · YOLO'));
    assert.equal(t.nodes.cameraStage.children[0].children.length,3);
    // The observer fires after hiding the captured preview to resume the existing stream.
    t.connect();const atResume=t.result.calls;t.nodes.startLiveGrowth.click();t.observer();
    await until(()=>t.result.calls>=atResume+2&&t.nodes.growthCameraStatus.textContent.includes('Live YOLO · frame'));
  }finally{t.cleanup();}
});
test('Recording, hidden tabs, and camera stop terminate live scanning',async()=>{
  const t=setup();try{
    t.connect();t.nodes.startLiveGrowth.click();await until(()=>t.result.calls>0);
    t.document.dispatchEvent({type:'bamboox:camera-state',detail:{active:true,busy:false,recording:true}});
    assert.equal(t.nodes.startLiveGrowth.disabled,true);assert.equal(t.nodes.stopLiveGrowth.disabled,true);
    await delay(30);t.document.dispatchEvent({type:'bamboox:camera-state',detail:{active:true,busy:false,recording:false}});
    t.nodes.startLiveGrowth.click();await until(()=>t.result.calls>1);
    t.document.hidden=true;t.document.dispatchEvent({type:'visibilitychange'});await delay(30);
    const calls=t.result.calls;await delay(60);assert.equal(t.result.calls,calls);assert.equal(t.nodes.startLiveGrowth.disabled,true);
    t.document.hidden=false;t.document.dispatchEvent({type:'visibilitychange'});t.nodes.startLiveGrowth.click();
    await until(()=>t.result.calls>calls);t.nodes.cameraVideo.srcObject=null;
    t.document.dispatchEvent({type:'bamboox:camera-state',detail:{active:false,busy:false,recording:false}});
    assert.equal(t.nodes.startLiveGrowth.disabled,true);assert.equal(t.nodes.cameraStage.children[0].style.display,'none');
  }finally{t.cleanup();}
});
