"""Full pre-registration -> local v14 residual -> current-raw application pipeline."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,hashlib,shutil,time,uuid,csv,io,copy,traceback,argparse,importlib.util
from pathlib import Path
import cv2,numpy as np
cv2.setNumThreads(2);cv2.ocl.setUseOpenCL(False)
ROOT=Path('D:/CODE/NR_models');HERE=ROOT/'coordination/v14_full_raw_current_20261024'
ART=ROOT/'artifacts/v14_full_raw_current_20261024';WORK=ART/'_work'
SYN=ART/'_synthetic_root';MAN=ROOT/'Train/manifest.json';CSV=ROOT/'coordination/resume_20261023/root_dataset_pairs.csv'
BULK=Path('C:/Users/thx11/Documents/Codex/NR_models_artifacts/20261024')
V5SRC=ROOT/'research/current_chat_flow_pyramid_v5_20261024/run.py'
V12SRC=ROOT/'research/current_chat_flow_boundary_v12_directional_20261024/run.py'
V14SRC=ROOT/'research/current_chat_foot_local_v14_20261024/multi_image.py'
UTIL=ROOT/'research/current_chat_flow_real_v2_detail_20261024/detail_residual_v2.py'
EVAL=ROOT/'experiments/alignment_pair2_20261023/generation/batch26m_current_20261023/evaluation/evaluator.py'
LOCK=ROOT/'research/regenerated_selection_20261024/integration/.apply.lock'
MODE='V14_FULL_PREREG_LOCAL_RESIDUAL_RAW_ONCE'
DEFER={63,64,69,70,90,106}
HIST_FULL={9:BULK/'research_current_chat_foot_local_v14/multi_image_v1/Train_9',18:BULK/'research_current_chat_foot_local_v14/multi_image_v1/Train_18',51:BULK/'research_current_chat_foot_local_v14/multi_image_v1/Train_51',209:BULK/'research_current_chat_foot_local_v14/multi_image_v1/Train_209',208:BULK/'research_current_chat_foot_local_v14/Train_208'}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
 with t.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def require(x,msg):
 if not x:raise RuntimeError(msg)
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def paths(i):return {k:ROOT/'Train'/k/f'{i}.png' for k in ('input','input_f','output','output_f')}
def support(m,w,h):return np.isfinite(m).all(2)&(m[...,0]>=0)&(m[...,0]<=w-1)&(m[...,1]>=0)&(m[...,1]<=h-1)
def jac(m):
 a=m[:-1,:-1];b=m[:-1,1:];c=m[1:,:-1];d=m[1:,1:];out=[]
 for dx,dy in ((b-a,c-a),(b-a,d-b),(d-c,c-a),(d-c,d-b)):out.append(float(np.min(dx[...,0]*dy[...,1]-dx[...,1]*dy[...,0])))
 return min(out)
def backup(src,dst):
 dst=Path(dst);require(not dst.exists(),'Backup exists '+str(dst));dst.parent.mkdir(parents=True,exist_ok=True)
 with Path(src).open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b,1048576);b.flush();os.fsync(b.fileno())
 require(sha(src)==sha(dst) and dst.stat().st_nlink==1,'Independent backup failed');return {'path':str(dst),'sha256':sha(dst)}
def replace(src,dst,old,new):
 require(sha(dst)==old,'CAS conflict '+str(dst));require(sha(src)==new,'Staged hash mismatch');t=dst.with_name(dst.name+'.full-v14-'+uuid.uuid4().hex+'.tmp')
 try:
  with Path(src).open('rb') as a,t.open('xb') as b:shutil.copyfileobj(a,b,1048576);b.flush();os.fsync(b.fileno())
  require(sha(t)==new and t.stat().st_nlink==1,'Temp validation failed');require(sha(dst)==old,'Concurrent modification');os.replace(t,dst);require(sha(dst)==new,'Replacement mismatch')
 finally:
  if t.exists():t.unlink()
def record(i):return next(r for r in read(MAN)['records'] if r['id']==i)
def load_map(p):
 p=Path(p)
 if p.suffix=='.npy':return np.load(p)
 z=np.load(p);return z['map_xy']
def save_map(p,m):
 np.savez_compressed(p,map_xy=m);require(np.array_equal(load_map(p),m),'Saved map replay mismatch')

def synthetic(i,rec,ps,hashes):
 """Create an uncorrected baseline from raw; never read the old adapter RGB for fitting."""
 ip=SYN/'Train/input_f'/f'{i}.png';tp=SYN/'Train/output_f'/f'{i}.png';ip.parent.mkdir(parents=True,exist_ok=True);tp.parent.mkdir(parents=True,exist_ok=True)
 if not ip.exists():os.link(ps['input_f'],ip)
 require(sha(ip)==hashes['input_f'],'Synthetic source binding mismatch')
 if tp.exists():require(sha(tp)==read(SYN/'Train/output_f'/f'{i}.json')['sha256'],'Synthetic baseline drift')
 else:
  raw=cv2.imread(str(ps['output']));A=np.array(rec['target_to_crop_affine'],np.float64)
  base=cv2.warpAffine(raw,A,(2000,1248),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT)
  cv2.imwrite(str(tp),base,[cv2.IMWRITE_PNG_COMPRESSION,3]);write(tp.with_suffix('.json'),{'id':i,'raw_sha256':hashes['output'],'affine':A.tolist(),'sha256':sha(tp),'source':'current raw once for independent uncorrected baseline; not old output_f'})
 return ip,tp

def historical_full(i,ps,hashes):
 src=HIST_FULL[i];m=load_map(src/'composite_source_to_target_float32.npy');res=read(src/'RESULT.json')
 require(res['input_hashes']['source']==hashes['input_f'] and res['input_hashes']['target']==hashes['output_f'],'Historical full-v14 source/target version mismatch')
 require(m.shape==(1248,2000,2) and m.dtype==np.float32 and np.isfinite(m).all(),'Historical map contract failed')
 # Historical candidate must replay exactly on the unchanged output_f before its map may be adapted.
 target=cv2.imread(str(ps['output_f']));old=cv2.remap(target,m[...,0],m[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
 require(np.array_equal(old,cv2.imread(str(src/'aligned_target_on_source.png'))),'Historical full-v14 PNG replay mismatch')
 return m,{'kind':'VERIFIED_HISTORICAL_FULL_V14','path':str(src),'map_sha256':sha(src/'composite_source_to_target_float32.npy'),'result_sha256':sha(src/'RESULT.json')}

def prereg(i,ps,hashes,rec,item):
 """Run frozen v5 global/local and v12 pre-registration on synthetic raw baseline."""
 ip,tp=synthetic(i,rec,ps,hashes);root=WORK/f'item_{i}';v5out=root/'v5';v12out=root/'v12';dst=v5out/f'Train_{i}';resume_v5=(dst/'FIT_RESULT.json').exists();require(resume_v5 or not dst.exists(),'Incomplete v5 stage requires explicit inspection');v5out.mkdir(parents=True,exist_ok=True);dst.mkdir(parents=True,exist_ok=True);v12out.mkdir(parents=True,exist_ok=True)
 oldpolicy=read(BULK/'research_current_chat_flow_pyramid_v5/POLICY.json');bound={str(V5SRC):sha(V5SRC),str(UTIL):sha(UTIL),str(EVAL):sha(EVAL),str(ip):sha(ip),str(tp):sha(tp)}
 policy={**oldpolicy,'ids':[i],'development':None,'fixed_new_image':[i],'bound_files':bound,'raw_baseline_sha256':hashes['output'],'old_adapter_output_f_used_for_fit':False}
 if not resume_v5:
  write(v5out/'POLICY.json',policy);write(v5out/'POLICY_FROZEN.json',{'sha256':sha(v5out/'POLICY.json'),'before_fit':True})
 else:
  require(read(v5out/'POLICY.json')==policy,'Resume policy mismatch')
 v5=load('frozen_v5_'+str(i),V5SRC);v5.ROOT=SYN;v5.OUT=v5out;v5.UTIL=UTIL
 if not resume_v5:
  for stage in ('global','flow','render'):
   t=time.monotonic();v5.worker(stage,i);write(dst/f'{stage}_FULL_CHAIN_RESOURCE.json',{'stage':stage,'seconds':time.monotonic()-t,'exit_code':0})
 v12=load('frozen_v12_'+str(i),V12SRC);v12.ROOT=SYN;v12.OUT=v12out;v12.BASE=v5out
 t=time.monotonic();v12.work(i);write(v12out/f'Train_{i}'/'FULL_CHAIN_RESOURCE.json',{'seconds':time.monotonic()-t,'exit_code':0})
 fit=read(v12out/f'Train_{i}'/'FIT_RESULT.json');require(fit['input_sha256']==hashes['input_f'] and fit['target_sha256']==sha(tp),'Prereg version binding mismatch')
 m=load_map(v12out/f'Train_{i}'/'composite_source_to_target_float32.npy')
 provenance={'kind':'NEW_FROZEN_V5_GLOBAL_LOCAL_PLUS_V12_PREREG','v5_dir':str(dst),'v12_dir':str(v12out/f'Train_{i}'),'v5_script_sha256':sha(V5SRC),'v12_script_sha256':sha(V12SRC),'v12_map_sha256':sha(v12out/f'Train_{i}'/'composite_source_to_target_float32.npy'),'synthetic_baseline_sha256':sha(tp),'precomponent_only_not_deliverable_v12':True}
 return m,v12out/f'Train_{i}'/'aligned_target_on_source.png',provenance

def residual_v14(m,parent,source,item):
 """Exact frozen multi_image.py lines 87-115 math, with full-canvas ROI for general items."""
 h,w=m.shape[:2];sg=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY);pg=cv2.cvtColor(parent,cv2.COLOR_BGR2GRAY);new=m.copy();flowinfo=[];changed_domain=np.zeros((h,w),bool)
 # Accepted full chain uses one full-crop local domain when no frozen character annotation exists.
 boxes=[(0,0,w,h)]
 for j,(a,b,c,d) in enumerate(boxes):
  x0=max(0,a-64);y0=max(0,b-64);x1=min(w,c+64);y1=min(h,d+64);s=sg[y0:y1,x0:x1].copy();p=pg[y0:y1,x0:x1].copy()
  def flow(a,b):
   dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM);dis.setFinestScale(0);dis.setPatchSize(8);dis.setPatchStride(3);dis.setGradientDescentIterations(25);dis.setVariationalRefinementIterations(5);return dis.calc(a,b,None)
  f=flow(s,p);r=flow(p,s);yy,xx=np.mgrid[:s.shape[0],:s.shape[1]].astype(np.float32);rx=xx+f[:,:,0];ry=yy+f[:,:,1]
  br=cv2.remap(r,rx,ry,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);cycle=np.linalg.norm(f+br,axis=2);length=np.linalg.norm(f,axis=2)
  grad=cv2.magnitude(cv2.Sobel(s,cv2.CV_32F,1,0),cv2.Sobel(s,cv2.CV_32F,0,1));known=(cycle<=.5)&(length<=4)&(grad>=4)&(rx>=2)&(rx<s.shape[1]-2)&(ry>=2)&(ry<s.shape[0]-2)
  confidence=known.astype(np.float32)*np.exp(-cycle**2/(2*.5**2));mass=cv2.GaussianBlur(confidence,(0,0),8);num=cv2.GaussianBlur(f*confidence[:,:,None],(0,0),8);field=num/(mass[:,:,None]+.05)
  X=xx+x0;Y=yy+y0;distance=np.sqrt(np.maximum(np.maximum(a-X,X-(c-1)),0)**2+np.maximum(np.maximum(b-Y,Y-(d-1)),0)**2);gate=np.clip(1-distance/64,0,1);gate=gate*gate*(3-2*gate);field*=gate[:,:,None]
  sx=X+field[:,:,0];sy=Y+field[:,:,1];bx=np.clip(np.floor(sx).astype(np.int32),0,w-2);by=np.clip(np.floor(sy).astype(np.int32),0,h-2);fx=(sx-bx)[:,:,None];fy=(sy-by)[:,:,None]
  comp=((1-fy)*((1-fx)*m[by,bx]+fx*m[by,bx+1])+fy*((1-fx)*m[by+1,bx]+fx*m[by+1,bx+1])).astype(np.float32)
  active=np.any(field!=0,axis=2);local=new[y0:y1,x0:x1];local[active]=comp[active];changed_domain[y0:y1,x0:x1]|=active
  flowinfo.append({'box':[a,b,c,d],'halo':[x0,y0,x1,y1],'solver_known':int(known.sum()),'solver_UNKNOWN':int((~known).sum()),'field_max_px':float(np.linalg.norm(field,axis=2).max())})
  np.savez_compressed(item/'v14_field_observations.npz',forward=f[::8,::8],reverse=r[::8,::8],cycle=cycle[::8,::8],known=known[::8,::8],field=field[::8,::8],gate=gate[::8,::8])
 require(not np.any((new!=m).any(2)&~changed_domain),'Residual changed outside local domain')
 return new,flowinfo

def panels(i,source,baseline,candidate,item,rawhash):
 from PIL import Image,ImageDraw
 ev=[];before=cv2.addWeighted(source,.5,baseline,.5,0);after=cv2.addWeighted(source,.5,candidate,.5,0)
 cv2.imwrite(str(item/'before_blend.png'),before);cv2.imwrite(str(item/'after_blend.png'),after)
 for n,(x,y) in enumerate(( (x,y) for y in (0,624) for x in (0,500,1000,1500) )):
  w=500;h=624;panel=Image.new('RGB',(1000,h+76),'#202020');d=ImageDraw.Draw(panel)
  d.text((4,4),f'Train {i} current raw SHA256 {rawhash}',fill='white');d.text((4,22),f'crop ROI [{x},{y},{x+w},{y+h}] native 1:1 alpha=.5',fill='white');d.text((4,40),'LEFT source+UNCORRECTED RAW BASELINE | RIGHT source+FULL V14',fill='white');d.text((4,58),'No source warp/fill/fallback; strict thresholds UNPROVEN',fill='white')
  for k,im in enumerate((before,after)):panel.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).crop((x,y,x+w,y+h)),(k*w,76))
  p=item/f'tile_{n:02d}_native_pair.png';panel.save(p);ev.append({'path':str(p),'sha256':sha(p),'crop_roi_xyxy':[x,y,x+w,y+h],'source_roi_xyxy':[x+104,y+24,x+w+104,y+h+24],'scale':1,'alpha':.5,'status':'NOT_VIEWED'})
 return ev

def prepare(i):
 require(i not in DEFER,'GENERATED_PENDING_SELECTION requires independent new-raw selection')
 require(shutil.disk_usage(ART).free>1536*1024**2,'DISK_BUDGET')
 item=ART/'items'/str(i);recover=(item/'ERROR.json').exists() and (item/'PREPARE_CHECKPOINT.json').exists() and not (item/'RECEIPT.json').exists();require(not item.exists() or recover,'Item exists; use verified recovery only');
 if not recover:item.mkdir(parents=True,exist_ok=False)
 t=time.monotonic();ps=paths(i);hashes={k:sha(p) for k,p in ps.items()};mh=sha(MAN);rec=record(i)
 require(all(hashes[k]==rec[k+'_sha256'] for k in hashes),'Manifest/live hash mismatch');require(np.array_equal(cv2.imread(str(ps['input']))[24:1272,104:2104],cv2.imread(str(ps['input_f']))),'Source crop mismatch')
 if recover:
  checkpoint=read(item/'PREPARE_CHECKPOINT.json');require(checkpoint['hashes']==hashes,'Recovery hash drift');backups=checkpoint['backups'];require(all(sha(v['path'])==v['sha256'] for v in backups.values()),'Recovery backup mismatch')
 else:
  backups={'output_f':backup(ps['output_f'],item/'backups/output_f.png'),'manifest':backup(MAN,item/'backups/manifest.json')}
  if CSV.exists():backups['csv']=backup(CSV,item/'backups/root_dataset_pairs.csv')
  write(item/'PREPARE_CHECKPOINT.json',{'id':i,'hashes':hashes,'backups':backups,'phase':'BACKUPS_VERIFIED'})
 raw=cv2.imread(str(ps['output']));source=cv2.imread(str(ps['input_f']));A=np.array(rec['target_to_crop_affine'],np.float64);Ai=np.linalg.inv(np.vstack([A,[0,0,1]]))[:2]
 baseline=cv2.warpAffine(raw,A,(2000,1248),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT)
 if i in HIST_FULL:
  m,prov=historical_full(i,ps,hashes);parent=None;flowinfo='FROZEN_HISTORICAL_FULL_MAP_REPLAYED'
 else:
  pre,parent_path,prov=prereg(i,ps,hashes,rec,item);parent=cv2.imread(str(parent_path));m,flowinfo=residual_v14(pre,parent,source,item);prov['v14_residual_script_sha256']=sha(V14SRC);prov['v14_residual_roi']='full canvas [0,0,2000,1248], halo clipped by canvas; same frozen DIS/gates/sigma/mass and map composition'
  np.save(item/'prereg_output_f_map.npy',pre)
 rawmap=(m@Ai[:,:2].T+Ai[:,2]).astype(np.float32);yy,xx=np.mgrid[:1248,:2000].astype(np.float32);grid=np.stack([xx,yy],2);basemap=(grid@Ai[:,:2].T+Ai[:,2]).astype(np.float32)
 sb=support(basemap,raw.shape[1],raw.shape[0]);sn=support(rawmap,raw.shape[1],raw.shape[0]);j=jac(rawmap)
 geom={'baseline_unsupported':int((~sb).sum()),'candidate_unsupported':int((~sn).sum()),'added_support_loss':int((sb&~sn).sum()),'saved_map_jacobian_min':j,'full_map_finite':bool(np.isfinite(rawmap).all()),'changed_map_pixels_vs_uncorrected_raw':int(np.any(rawmap!=basemap,2).sum()),'map_domain':'fixed_source_crop_to_current_raw_output'}
 gate=geom['added_support_loss']==0 and j>=.05 and geom['full_map_finite'];geom['gate_pass']=gate
 candidate=cv2.remap(raw,rawmap[...,0],rawmap[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);cp=item/'candidate_output_f.png';cv2.imwrite(str(cp),candidate,[cv2.IMWRITE_PNG_COMPRESSION,3]);mp=item/'source_crop_to_current_raw_float32.npz';save_map(mp,rawmap)
 require(np.array_equal(cv2.imread(str(cp)),cv2.remap(raw,load_map(mp)[...,0],load_map(mp)[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)),'Candidate map replay mismatch')
 ev=panels(i,source,baseline,candidate,item,hashes['output']);require(all(sha(ps[k])==v for k,v in hashes.items()),'Train drift during prepare')
 status='PREPARED_NOT_VIEWED' if gate else 'HELD_SUPPORT'
 if geom['changed_map_pixels_vs_uncorrected_raw']==0:status='NO_EFFECT'
 r={'id':i,'status':status,'reason':None if gate else 'Full-chain raw support/fold gate failed; no clipping/fill/fallback; not a visible-failure decision.','hashes_before':hashes,'paths':{k:str(p) for k,p in ps.items()},'prior_record':rec,'manifest_sha256_prepare':mh,'backups':backups,'pipeline':MODE,'pipeline_script_sha256':sha(__file__),'preregistration':prov,'v14_flow':flowinfo,'geometry':geom,'gate_pass':gate,'candidate_path':str(cp),'candidate_sha256':sha(cp),'map_path':str(mp),'map_sha256':sha(mp),'map_replay_pixel_exact':True,'raw_sampling_calls_for_candidate':1,'baseline_source':'current raw + manifest affine; old output_f RGB not fitted or resampled','old_adapter_output_f_sha256':hashes['output_f'],'evidence':ev,'visual_status':'NOT_VIEWED','strict_boundary_certification':'UNPROVEN','fallback_used':False,'source_warped':False,'source_fill':False,'seconds':time.monotonic()-t,'train_writes':False}
 write(item/'RECEIPT.json',r);print(json.dumps({'id':i,'status':status,'geometry':geom,'seconds':r['seconds']}),flush=True);return r

def review(i,decision,note):
 rp=ART/'items'/str(i)/'RECEIPT.json';r=read(rp);require(r['status'] in ('PREPARED_NOT_VIEWED','HELD_VISIBLE'),'Cannot review state');require(decision in ('ACCEPT','HOLD'),'Bad decision')
 for e in r['evidence']:require(sha(e['path'])==e['sha256'],'Evidence drift');e['status']='VIEWED_NATIVE_50_PERCENT'
 r.update(status='REVIEWED_ACCEPT' if decision=='ACCEPT' else 'HELD_VISIBLE',visual_status='NO_OBVIOUS_REGRESSION_IN_VIEWED_ROIS' if decision=='ACCEPT' else 'FAIL_VISIBLE',review_note=note,whole_image_visual_pass=False);write(rp,r)

def journal(item,x):
 with (item/'JOURNAL.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def apply(i):
 item=ART/'items'/str(i);rp=item/'RECEIPT.json';r=read(rp);require(r['status']=='REVIEWED_ACCEPT' and r['gate_pass'],'Requires actual ACCEPT and gate pass');ps=paths(i);require(all(sha(ps[k])==v for k,v in r['hashes_before'].items()),'Dataset drift');require(sha(r['candidate_path'])==r['candidate_sha256'] and sha(r['map_path'])==r['map_sha256'],'Candidate drift')
 LOCK.parent.mkdir(parents=True,exist_ok=True);fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(item).encode());os.close(fd);ops=[];done=[]
 try:
  require(not (item/'JOURNAL.jsonl').exists(),'Prior transaction needs recovery');mh=sha(MAN);m=read(MAN);rec=next(x for x in m['records'] if x['id']==i);require(rec==r['prior_record'],'Manifest record changed')
  mb=item/'commit_backups/manifest.json';backup(MAN,mb);rec['v14_full_chain_prior_record']=copy.deepcopy(rec)
  rec.update(output_f_sha256=r['candidate_sha256'],processing_mode=MODE,local_warp_used=True,raw_target_sampling_calls=1,source_output_f=r['candidate_path'],active_raw_sampling_map=r['map_path'],active_raw_sampling_map_sha256=r['map_sha256'],active_raw_sampling_map_direction='fixed_source_crop_to_raw_output',source_report=str(rp),source_report_sha256=sha(rp),v14_core_applied=True,v14_full_chain_applied=True,v14_preregistration=r['preregistration'],v14_geometry=r['geometry'],v14_visual_status=r['visual_status'],v14_fallback_used=False,geometry_acceptance='VIEWED_ROIS_NO_OBVIOUS_REGRESSION_STRICT_UNPROVEN',training_ready=False,needsreview=True,review_reason='Full pre-registration + v14 local residual applied from current raw once; native listed tiles reviewed; strict thresholds UNPROVEN')
  pm=item/'manifest.PROPOSED.json';write(pm,m);ops=[{'dest':str(ps['output_f']),'src':r['candidate_path'],'old':r['hashes_before']['output_f'],'new':r['candidate_sha256'],'backup':r['backups']['output_f']['path']},{'dest':str(MAN),'src':str(pm),'old':mh,'new':sha(pm),'backup':str(mb)}]
  if CSV.exists():
   ch=sha(CSV);cb=item/'commit_backups/root_dataset_pairs.csv';backup(CSV,cb);reader=csv.DictReader(io.StringIO(CSV.read_text(encoding='utf-8-sig')));fields=list(reader.fieldnames);rows=list(reader);matches=[x for x in rows if x['split']=='Train' and int(x['id'])==i];require(len(matches)==1,'CSV ID mismatch');row=matches[0];require(row['output_f_sha256']==r['hashes_before']['output_f'] and row['output_sha256']==r['hashes_before']['output'],'CSV drift')
   for k in ('output_f_sha256','processing_mode','review_reason'):row[k]=rec[k]
   row.update(active_raw_sampling_map=r['map_path'],active_raw_sampling_map_sha256=r['map_sha256'],v14_application_receipt=str(rp),strict_boundary_certification='UNPROVEN',local_warp_used='True',visual_status=r['visual_status'],training_ready='False');pc=item/'root_dataset_pairs.PROPOSED.csv'
   with pc.open('x',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
   ops.append({'dest':str(CSV),'src':str(pc),'old':ch,'new':sha(pc),'backup':str(cb)})
  write(item/'PLAN.json',{'id':i,'operations':ops,'raw_and_input_writes':False});journal(item,{'phase':'BEGIN','id':i})
  for o in ops:
   require(sha(o['backup'])==o['old'],'Commit backup mismatch');journal(item,{'phase':'INTENT',**o});done.append(o);replace(Path(o['src']),Path(o['dest']),o['old'],o['new']);journal(item,{'phase':'REPLACED','dest':o['dest']})
  require(all(sha(ps[k])==r['hashes_before'][k] for k in ('input','input_f','output')),'Protected file changed');journal(item,{'phase':'COMMIT'})
  result={'id':i,'status':'APPLIED_FULL_V14','output_f_sha256':sha(ps['output_f']),'map_sha256':r['map_sha256'],'receipt_sha256':sha(rp),'manifest_sha256_after':sha(MAN),'protected_files_unchanged':True,'backup_verified':True,'fallback_used':False,'strict_boundary_certification':'UNPROVEN','operations':ops};write(item/'APPLY_RECEIPT.json',result);print(json.dumps({'id':i,'status':result['status']}),flush=True)
 except Exception as e:
  errors=[]
  for o in reversed(done):
   try:
    cur=sha(o['dest'])
    if cur==o['old']:continue
    require(cur==o['new'],'Rollback CAS conflict');replace(Path(o['backup']),Path(o['dest']),cur,o['old'])
   except Exception as err:errors.append(str(err))
  write(item/'APPLY_ERROR.json',{'status':'ERROR','error':str(e),'rollback_errors':errors});raise
 finally:LOCK.unlink()

def summary():
 rows=[]
 for rec in read(MAN)['records']:
  i=rec['id'];item=ART/'items'/str(i)
  if i in DEFER:st='GENERATED_PENDING_SELECTION'
  elif (item/'APPLY_RECEIPT.json').exists():st='APPLIED_FULL_V14'
  elif (item/'RECEIPT.json').exists():st=read(item/'RECEIPT.json')['status']
  else:st='PENDING_FULL_CHAIN'
  rows.append({'id':i,'status':st,'manifest_mode':rec['processing_mode'],'local_warp_used':rec.get('local_warp_used')})
 counts={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})};write(HERE/'STATUS.json',{'counts':counts,'total':len(rows),'items':rows,'strict_boundary_certification':'UNPROVEN'});print(json.dumps(counts),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','review','apply','summary']);p.add_argument('--ids',nargs='+',type=int);p.add_argument('--decision');p.add_argument('--note',default='');a=p.parse_args()
 if a.command=='summary':summary();return
 require(a.ids and len(a.ids)<=20,'Batch size must be 1..20')
 for i in a.ids:
  try:
   if a.command=='prepare':prepare(i)
   elif a.command=='review':review(i,a.decision,a.note)
   else:apply(i)
  except Exception as e:
   item=ART/'items'/str(i)
   # A sealed receipt is terminal preparation evidence. A repeated command must
   # not overwrite its status with an unrelated invocation error.
   if (item/'RECEIPT.json').exists():
    print(json.dumps({'id':i,'status':'TERMINAL_RECEIPT_RETAINED','error':str(e)}),flush=True)
   else:
    item.mkdir(parents=True,exist_ok=True);write(item/'ERROR.json',{'id':i,'status':'ERROR','error':str(e),'traceback':traceback.format_exc()});print(json.dumps({'id':i,'status':'ERROR','error':str(e)}),flush=True)
 summary()
if __name__=='__main__':main()
