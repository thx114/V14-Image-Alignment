import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,json,time,hashlib,importlib.util,subprocess
ROOT=Path('D:/CODE/NR_models'); HERE=Path(__file__).resolve().parent
OUT=Path('C:/Users/thx11/Documents/Codex/NR_models_artifacts/20261024/research_current_chat_flow_boundary_v12_directional')
BASE=OUT.parent/'research_current_chat_flow_v5_hardcases'
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def work(ident):
 import cv2,numpy as np,psutil
 psutil.Process().cpu_affinity(psutil.Process().cpu_affinity()[:2]);cv2.setNumThreads(2);cv2.ocl.setUseOpenCL(False)
 spec=importlib.util.spec_from_file_location('utils',ROOT/'research/current_chat_flow_real_v2_detail_20261024/detail_residual_v2.py');u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
 dst=OUT/f'Train_{ident}';dst.mkdir(parents=True)
 old=BASE/f'Train_{ident}'; A=np.load(old/'global.npz')['affine']; z=np.load(old/'correspondences_for_26m.npz'); h,w=1248,2000
 yy,xx=np.mgrid[:h,:w];grid=np.stack([xx,yy],-1).astype(np.float32);del xx,yy
 xy=z['source_xy']; k=z['known']; ix,iy=xy[k].astype(int).T; weights=z['weight'][k]
 mass=np.zeros((h,w),np.float32);num=np.zeros((h,w,2),np.float32);seed=np.ones((h,w),np.uint8)
 mass[iy,ix]=weights;num[iy,ix]=z['residual_source_px'][k]*weights[:,None];seed[iy,ix]=0
 mass=cv2.GaussianBlur(mass,(385,385),32,borderType=cv2.BORDER_REFLECT_101);num=cv2.GaussianBlur(num,(385,385),32,borderType=cv2.BORDER_REFLECT_101)
 distance=cv2.distanceTransform(seed,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
 gate=np.clip((128-distance)/64,0,1);gate=gate*gate*(3-2*gate)
 # Smooth mass regularization, no hard cutoff and no source-border zero band.
 velocity=(num/(mass[:,:,None]+1e-7)*gate[:,:,None]).astype(np.float32)
 del mass,num,distance,gate
 # Directional boundary constraint in target coordinates; retain inward motion.
 base_target=grid@A[:,:2].T+A[:,2]
 target_velocity=velocity@A[:,:2].T
 for axis,limit in ((0,w-1),(1,h-1)):
  inward_distance=np.where(target_velocity[:,:,axis]<0,base_target[:,:,axis],limit-base_target[:,:,axis])
  fraction=np.clip(inward_distance/64,0,1)
  target_velocity[:,:,axis]*=fraction*fraction*(3-2*fraction)
 velocity=(target_velocity@np.linalg.inv(A[:,:2]).T).astype(np.float32)
 del target_velocity,base_target,inward_distance,fraction
 # A fixed SVF interpretation is a NEW fit-only geometric candidate, not an exact repair.
 displacement=velocity/64
 integration=[]
 for i in range(6):
  g=grid+displacement
  from scipy.ndimage import map_coordinates
  sampled=np.stack([map_coordinates(displacement[:,:,c],[g[:,:,1],g[:,:,0]],order=1,mode='nearest',prefilter=False) for c in range(2)],-1)
  displacement+=sampled
  integration.append({'squaring':i+1,'max_displacement_px':float(np.linalg.norm(displacement,axis=2).max())})
 local=grid+displacement;mapping=(local@A[:,:2].T+A[:,2]).astype(np.float32)
 base=(grid@A[:,:2].T+A[:,2]).astype(np.float32)
 def valid(m):return np.isfinite(m).all(2)&(m[:,:,0]>=0)&(m[:,:,0]<=w-1)&(m[:,:,1]>=0)&(m[:,:,1]<=h-1)
 good=valid(mapping);gb=valid(base);det=u.jacobian_min(mapping)
 target=cv2.imread(str(ROOT/f'Train/output_f/{ident}.png'));src=cv2.imread(str(ROOT/f'Train/input_f/{ident}.png'))
 candidate=cv2.remap(target,mapping[:,:,0],mapping[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
 np.save(dst/'composite_source_to_target_float32.npy',mapping);cv2.imwrite(str(dst/'aligned_target_on_source.png'),candidate)
 report={'id':ident,'n_squarings':6,'alpha':1,'integration':integration,'jacobian_min':det,'fold_check_pass':bool(det>=.05),'full_support_loss':int((~good).sum()),'baseline_support_loss':int((~gb).sum()),'new_support_loss_pixels':int((gb&~good).sum()),'map_sha256':sha(dst/'composite_source_to_target_float32.npy'),'png_sha256':sha(dst/'aligned_target_on_source.png'),'input_sha256':sha(ROOT/f'Train/input_f/{ident}.png'),'target_sha256':sha(ROOT/f'Train/output_f/{ident}.png'),'no_eval_used_for_fit':True,'gt':False,'global_injectivity_proven':False,'character_edges_proven':False,'render':'original color target once INTER_LINEAR, no filtering or sharpening'}
 write(dst/'FIT_RESULT.json',report)
 q=u.query_grid(w,h);bk,be=u.independent_eval(cv2.imread(str(ROOT/f'Train/input_f/{ident}.png'),0),cv2.imread(str(ROOT/f'Train/output_f/{ident}.png'),0),q);ak,ae=u.independent_eval(cv2.imread(str(ROOT/f'Train/input_f/{ident}.png'),0),cv2.imread(str(dst/'aligned_target_on_source.png'),0),q);common=bk&ak
 np.savez_compressed(dst/'independent_query_records.npz',query_xy=q,before_known=bk,after_known=ak,before_error=be,after_error=ae)
 write(dst/'EVALUATION.json',{'denominator':len(q),'before_known':int(bk.sum()),'after_known':int(ak.sum()),'lost':int((bk&~ak).sum()),'gained':int((~bk&ak).sum()),'before_common':u.stat(be[common]),'after_common':u.stat(ae[common]),'after_all_known':u.stat(ae[ak]),'not_GT':True,'full_target_proven':False})
 print(json.dumps(report))
if __name__=='__main__':
 if len(sys.argv)>1:work(int(sys.argv[1]))
 else:
  import psutil
  OUT.mkdir(parents=True,exist_ok=True)
  write(OUT/'POLICY_FROZEN.json',{'ids':[208],'interpretation':'v5 frozen reliable observations; Gaussian sigma32 kernel385 normalized weights; extend finite Gaussian truncation from3sigma to6sigma plus1e-7 mass prior, support smooth64..128px, directional target boundary attenuation smooth0to64px outward components only, inward retained, SVF n6 exact scipy linear interpolation nearest boundary alpha1; replaces only cv2 table interpolation during integration; hypothesis quantization introduces artificial map discontinuity; pre-fixed no eval selection','script_sha256':sha(Path(__file__)),'accept':'finite cell corner determinants>=.05 and zero added affine support loss; pixel support is not correspondence confidence','visual':'unfiltered RGB target once sampled; inspect source/target50percent overlays','status':'research diagnostic; not dense GT, not global injectivity proof'})
  runs=[]
  for ident in [208]:
   start=time.monotonic();p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),str(ident)],stdout=(OUT/f'{ident}.stdout').open('w'),stderr=(OUT/f'{ident}.stderr').open('w'));peak=0;reason=None
   while p.poll() is None:
    try:peak=max(peak,psutil.Process(p.pid).memory_info().private/1048576)
    except psutil.Error:pass
    if peak>900 or time.monotonic()-start>85: reason='resource_limit';p.kill();break
    time.sleep(.05)
   p.wait();runs.append({'id':ident,'exit_code':p.returncode,'sampled_private_mib':peak,'seconds':time.monotonic()-start,'limit_failure':reason})
  write(OUT/'RESOURCE_RECEIPT.json',runs);print(json.dumps(runs))
