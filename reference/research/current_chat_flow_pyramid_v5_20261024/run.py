import os
os.environ.update({k:'2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,hashlib,subprocess,importlib.util
from pathlib import Path
import psutil
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=Path('C:/Users/thx11/Documents/Codex/NR_models_artifacts/20261024/research_current_chat_flow_pyramid_v5')
UTIL=ROOT/'research/current_chat_flow_real_v2_detail_20261024/detail_residual_v2.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,indent=2,allow_nan=False)
def worker(stage,ident):
 import cv2,numpy as np
 cv2.setNumThreads(2);cv2.ocl.setUseOpenCL(False);cv2.setRNGSeed(20261024)
 spec=importlib.util.spec_from_file_location('utility',UTIL);u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
 policy=json.loads((OUT/'POLICY.json').read_text())
 for name,hsh in policy['bound_files'].items():assert sha(name)==hsh,name
 dst=OUT/f'Train_{ident}';ip=ROOT/f'Train/input_f/{ident}.png';tp=ROOT/f'Train/output_f/{ident}.png'
 src=cv2.imread(str(ip),0);target=cv2.imread(str(tp),0);h,w=src.shape
 yy,xx=np.mgrid[:h,:w].astype(np.float32);grid=np.stack((xx,yy),2)
 def sample(a,p):return cv2.remap(a,p[:,0].astype(np.float32).reshape(-1,1),p[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(len(p),-1).squeeze()
 def valid(p):return np.isfinite(p).all(-1)&(p[...,0]>=0)&(p[...,0]<=w-1)&(p[...,1]>=0)&(p[...,1]<=h-1)
 if stage=='global':
  sift=cv2.SIFT_create(nfeatures=12000);ka,da=sift.detectAndCompute(src,None);kb,db=sift.detectAndCompute(target,None)
  matches=cv2.BFMatcher().knnMatch(da,db,k=2);pairs=[(ka[a.queryIdx].pt,kb[a.trainIdx].pt) for a,b in matches if a.distance<.72*b.distance]
  pairs=np.array(pairs,np.float32);p=pairs[:,0];q=pairs[:,1]
  # Exclude a square around each independently frozen eval query from global fitting.
  rem=(p-80)%32;near=np.minimum(rem,32-rem);keep=~((near[:,0]<8)&(near[:,1]<8));p=p[keep];q=q[keep]
  A,inliers=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=10000,confidence=.995)
  assert A is not None and inliers.sum()>=12
  np.savez_compressed(dst/'global.npz',affine=A,source_xy=p,target_xy=q,inlier=inliers[:,0].astype(bool))
  write(dst/'GLOBAL.json',{'matches':len(p),'inliers':int(inliers.sum()),'affine':A.tolist(),'source_sha256':sha(ip),'target_sha256':sha(tp)})
 elif stage=='flow':
  A=np.load(dst/'global.npz')['affine'];base=(grid@A[:,:2].T+A[:,2]).astype(np.float32)
  pulled=cv2.remap(target,base[:,:,0],base[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
  clahe=cv2.createCLAHE(clipLimit=2,tileGridSize=(16,16));s=clahe.apply(src);t=clahe.apply(pulled)
  forward=backward=None;timing=[]
  for scale in policy['scales']:
   start=time.monotonic();nw=int(round(w*scale));nh=int(round(h*scale));ss=cv2.resize(s,(nw,nh),interpolation=cv2.INTER_AREA);tt=cv2.resize(t,(nw,nh),interpolation=cv2.INTER_AREA)
   if forward is not None:
    oh,ow=forward.shape[:2];factor=np.array([nw/ow,nh/oh],np.float32);forward=cv2.resize(forward,(nw,nh))*factor;backward=cv2.resize(backward,(nw,nh))*factor
   dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM);dis.setFinestScale(0);dis.setUseSpatialPropagation(True)
   forward=dis.calc(ss,tt,forward);backward=dis.calc(tt,ss,backward);timing.append({'scale':scale,'seconds':time.monotonic()-start})
  xy=np.array([(x,y) for y in range(72,h-64,16) for x in range(72,w-64,16)],np.float32)
  flow=sample(forward,xy);end=xy+flow;rev=sample(backward,end);cycle=np.linalg.norm(flow+rev,axis=1)
  sf=src.astype(np.float32);texture=np.sqrt(np.maximum(cv2.blur(sf*sf,(9,9))-cv2.blur(sf,(9,9))**2,0));tex=sample(texture,xy)
  median=np.dstack([cv2.medianBlur(forward[:,:,i],5) for i in range(2)]);consistency=np.linalg.norm(flow-sample(median,xy),axis=1)
  raw=end@A[:,:2].T+A[:,2];known=valid(end)&valid(raw)&valid(xy@A[:,:2].T+A[:,2])&(cycle<=1.5)&(tex>=3)&(np.linalg.norm(flow,axis=1)<=24)&(consistency<=2.5)
  edge=np.abs(cv2.Laplacian(sf,cv2.CV_32F));edge/=max(float(np.percentile(edge,95)),1);weight=np.exp(-(cycle/1.5)**2)*(.15+.85*np.minimum(sample(edge,xy),1));weight=np.where(known,weight,0).astype(np.float32)
  np.savez_compressed(dst/'correspondences_for_26m.npz',source_xy=xy,raw_target_xy=raw.astype(np.float32),weight=weight,known=known,cycle_px=cycle,residual_source_px=flow,affine_source_to_raw_target=A,texture_std=tex,local_consistency_px=consistency)
  write(dst/'FLOW.json',{'fit_denominator':len(xy),'known':int(known.sum()),'unknown':int((~known).sum()),'coverage':float(known.mean()),'cycle_gate_px':1.5,'known_residual_p50_p95_max':np.percentile(np.linalg.norm(flow[known],axis=1),[50,95,100]).tolist() if known.any() else [],'scales':timing,'note':'cycle and texture are reliability proxies, not GT; only one row per source, no duplicated global constraint'})
 elif stage=='render':
  z=np.load(dst/'correspondences_for_26m.npz');A=z['affine_source_to_raw_target'];xy=z['source_xy'];known=z['known'];assert known.sum()>=12,'INSUFFICIENT_RELIABLE_FIT'
  mass=np.zeros((h,w),np.float32);num=np.zeros((h,w,2),np.float32);seed=np.ones((h,w),np.uint8)
  ix=xy[known,0].astype(int);iy=xy[known,1].astype(int);weight=z['weight'][known];mass[iy,ix]=weight;num[iy,ix]=z['residual_source_px'][known]*weight[:,None];seed[iy,ix]=0
  mass=cv2.GaussianBlur(mass,(145,145),24,borderType=cv2.BORDER_CONSTANT);num=cv2.GaussianBlur(num,(145,145),24,borderType=cv2.BORDER_CONSTANT)
  dist=cv2.distanceTransform(seed,cv2.DIST_L2,cv2.DIST_MASK_PRECISE);gate=np.clip((40-dist)/16,0,1);gate=gate*gate*(3-2*gate);gate*=mass>=1e-6
  border=np.minimum.reduce([xx,yy,w-1-xx,h-1-yy]);taper=np.clip((border-64)/32,0,1);taper=taper*taper*(3-2*taper)
  field=num/np.maximum(mass[:,:,None],1e-6)*(gate*taper)[:,:,None];base=(grid@A[:,:2].T+A[:,2]).astype(np.float32);mapping=(base+field@A[:,:2].T).astype(np.float32)
  det=u.jacobian_min(mapping);sb=valid(base);sc=valid(mapping);delta=np.linalg.norm(mapping-base,axis=2)
  np.save(dst/'composite_source_to_target_float32.npy',mapping);color=cv2.imread(str(tp));candidate=cv2.remap(color,mapping[:,:,0],mapping[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);cv2.imwrite(str(dst/'aligned_target_on_source.png'),candidate)
  np.savez_compressed(dst/'support_confidence.npz',map_in_bounds=sc,affine_in_bounds=sb,residual_influence=gate*taper,interpolation_mass=mass)
  write(dst/'FIT_RESULT.json',{'status':'DIAGNOSTIC_REVIEW' if det>=.05 and not (sb&~sc).any() else 'REJECTED_GEOMETRY_OR_SUPPORT','jacobian_min_all_cell_corners':det,'no_fold_local_only':bool(det>=.05),'full_support_loss':int((~sc).sum()),'affine_support_loss':int((~sb).sum()),'added_support_loss':int((sb&~sc).sum()),'delta_from_affine_p50_p95_p99_max':np.percentile(delta,[50,95,99,100]).tolist(),'nonzero_local_pixels':int((delta>1e-6).sum()),'semantic_character_protection':'UNKNOWN','global_injectivity':'UNKNOWN','alpha':1,'sampling':'original color target once, INTER_LINEAR, zero outside; no coordinate clipping'})
  files=['composite_source_to_target_float32.npy','aligned_target_on_source.png','correspondences_for_26m.npz','support_confidence.npz','FIT_RESULT.json']
  write(dst/'CANDIDATE_FROZEN_BEFORE_EVAL.json',{'files':{f:sha(dst/f) for f in files},'policy_sha256':sha(OUT/'POLICY.json'),'input_sha256':sha(ip),'target_sha256':sha(tp)})
 elif stage=='eval':
  frozen=json.loads((dst/'CANDIDATE_FROZEN_BEFORE_EVAL.json').read_text())
  for f,hsh in frozen['files'].items():assert sha(dst/f)==hsh
  mapping=np.load(dst/'composite_source_to_target_float32.npy');disk=cv2.imread(str(dst/'aligned_target_on_source.png'));replay=cv2.remap(cv2.imread(str(tp)),mapping[:,:,0],mapping[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);assert np.array_equal(disk,replay)
  A=np.load(dst/'global.npz')['affine'];base=(grid@A[:,:2].T+A[:,2]).astype(np.float32);affine=cv2.remap(target,base[:,:,0],base[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);after=cv2.cvtColor(disk,cv2.COLOR_BGR2GRAY);q=u.query_grid(w,h)
  bk,be=u.independent_eval(src,target,q);gk,ge=u.independent_eval(src,affine,q);ak,ae=u.independent_eval(src,after,q);common=bk&gk&ak
  np.savez_compressed(dst/'independent_query_records.npz',source_query_xy=q,before_known=bk,affine_known=gk,after_known=ak,before_error=be,affine_error=ge,after_error=ae)
  tail=bk&(be>=np.percentile(be[bk],95));common2=bk&ak
  result={'denominator':len(q),'before_known':int(bk.sum()),'affine_known':int(gk.sum()),'after_known':int(ak.sum()),'before_unknown':int((~bk).sum()),'after_unknown':int((~ak).sum()),'lost_known':int((bk&~ak).sum()),'gained_known':int((~bk&ak).sum()),'three_way_common':int(common.sum()),'before_common':u.stat(be[common]),'affine_common':u.stat(ge[common]),'after_common':u.stat(ae[common]),'before_pair_common':u.stat(be[common2]),'after_pair_common':u.stat(ae[common2]),'after_all_known':u.stat(ae[ak]),'before_tail_count':int(tail.sum()),'before_tail_lost':int((tail&~ak).sum()),'before_tail_retained_after':u.stat(ae[tail&ak]),'tail_map_delta_from_affine':u.stat(np.linalg.norm(sample(mapping-base,q[tail]),axis=1)),'png_replay_different_channels':int((disk!=replay).sum()),'map_jacobian_recheck':u.jacobian_min(mapping),'full_image_target_proven':False,'character_edge_target_proven':False,'character_edge_correspondence':'UNKNOWN','independent_estimator':'fixed-grid DoG NCC with independent reverse search; not GT','input_sha256':sha(ip),'target_sha256':sha(tp)}
  write(dst/'RESULT.json',result)
  # Diagnostic images preserve original samples; no image blur to hide misalignment.
  source_color=cv2.imread(str(ip));small=lambda a:cv2.resize(a,(1000,624),interpolation=cv2.INTER_AREA)
  panels=[small(source_color),small(cv2.imread(str(tp))),small(disk)]
  for panel,label in zip(panels,['SOURCE','CURRENT ROOT TARGET','PYRAMID FLOW CANDIDATE']):cv2.putText(panel,label,(20,32),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
  cv2.imwrite(str(dst/'comparison.png'),np.hstack(panels))
def main():
 if len(sys.argv)>1:return worker(sys.argv[1],int(sys.argv[2]))
 OUT.mkdir(parents=True,exist_ok=False)
 files=[Path(__file__),UTIL,ROOT/'experiments/alignment_pair2_20261023/generation/batch26m_current_20261023/evaluation/evaluator.py']+[ROOT/f'Train/{kind}/{i}.png' for i in [2,8] for kind in ['input_f','output_f']]
 policy={'version':'pyramid-v5','ids':[2,8],'development':2,'fixed_new_image':8,'fit_grid':'margin72 step16; excludes eval centers by8px per axis','eval_grid':'margin80 step32; evaluator unchanged','global':'SIFT12000 ratio.72 partialAffine RANSAC2px 10000iter; spatial eval exclusion8px','scales':[.125,.25,.5,1.0],'DIS':'MEDIUM finestScale0 spatial propagation; resized previous flow initialized at every scale','representation':'CLAHE clip2 tiles16x16, output remains unfiltered original color target','cycle_gate_px':1.5,'texture_std_min':3,'residual_max_px':24,'local_median_consistency_max_px':2.5,'weight':'exp(-(cycle/1.5)^2) * (.15+.85*edge)','interpolation':'Gaussian(weight*residual)/Gaussian(weight), sigma24 kernel145; support40 smoothstep24to40; border64to96','alpha':1,'geometry':'reject det<.05 or additional support loss; preserve rejected map; no alpha search','coordinates':'native current-root source to current-root target; x column y row integer centers','exports':'one flow correspondence per source row; global affine provenance separate, never double count','goals':{'full_image_strict_less_px':2,'character_edge_strict_less_px':1.5,'semantic_protection':'UNKNOWN; no access to parent contour eval'},'resources':{'cpu':2,'gpu':0,'child_seconds':85,'private_mib':900},'bound_files':{str(f):sha(f) for f in files}}
 write(OUT/'POLICY.json',policy);write(HERE/'POLICY.json',policy);write(OUT/'POLICY_FROZEN.json',{'sha256':sha(OUT/'POLICY.json'),'before_fit':True});write(HERE/'STATUS_STARTED.json',{'status':'FROZEN_BEFORE_FIT','bulk':str(OUT)})
 receipts=[]
 for ident in [2,8]:
  (OUT/f'Train_{ident}').mkdir()
  for stage in ['global','flow','render','eval']:
   start=time.monotonic();peak=0;reason=None
   with (OUT/f'Train_{ident}'/(stage+'.log')).open('x') as log:
    child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),stage,str(ident)],stdout=log,stderr=subprocess.STDOUT);proc=psutil.Process(child.pid);proc.cpu_affinity(proc.cpu_affinity()[:2]);aff=proc.cpu_affinity()
    while child.poll() is None:
     try:
      info=proc.memory_info();peak=max(peak,getattr(info,'private',info.rss))
      if time.monotonic()-start>85 or peak>900*1048576:reason='TIME_OR_MEMORY_LIMIT';child.kill();break
     except psutil.NoSuchProcess:break
     time.sleep(.05)
    code=child.wait()
   receipt={'id':ident,'stage':stage,'exit_code':code,'seconds':time.monotonic()-start,'sampled_private_mib':peak/1048576,'affinity':aff,'limit_failure':reason};receipts.append(receipt);write(OUT/f'Train_{ident}'/(stage+'_RESOURCE.json'),receipt);print(json.dumps(receipt),flush=True)
   if code or reason:break
 write(HERE/'STATUS_FINISHED.json',{'status':'EXPERIMENT_FINISHED_CHECK_RESULTS','receipts':receipts,'bulk':str(OUT),'training_modified':False})
if __name__=='__main__':main()
