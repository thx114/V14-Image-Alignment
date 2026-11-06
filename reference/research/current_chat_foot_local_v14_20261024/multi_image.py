"""Frozen, bounded v14 transfer to four source-annotated character layouts."""
import os,sys,json,time,hashlib,subprocess
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
from pathlib import Path
import psutil
ROOT=Path('D:/CODE/NR_models');HERE=Path(__file__).resolve().parent
BULK=Path('C:/Users/thx11/Documents/Codex/NR_models_artifacts/20261024')
OUT=BULK/'research_current_chat_foot_local_v14/multi_image_v1'
IDS=[9,51,18,209]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def rows():return {r['id']:r['source_character_rectangles'] for r in json.loads((ROOT/'coordination/v5_hardcase_fold_locations_20261024.json').read_text())['rows']}
def v5(i):return BULK/('research_current_chat_flow_v5_hardcases_side' if i in (9,51) else 'research_current_chat_flow_v5_hardcases')/f'Train_{i}'
def v12(i):return (BULK/'research_current_chat_flow_boundary_v12_hardcases_side' if i in (9,51) else OUT/'v12_generated')/f'Train_{i}'
def setup():
    if (OUT/'POLICY_FROZEN.json').exists():raise RuntimeError('Frozen multi-image run already exists')
    OUT.mkdir(parents=True,exist_ok=True)
    policy={'ids':IDS,'source_rectangles':{str(i):rows()[i] for i in IDS},'annotation_sha256':sha(ROOT/'coordination/v5_hardcase_fold_locations_20261024.json'),'roi_semantics':'source-only rectangles include background; not semantic contours; all four 209 people retained','flow':'same v14 fixed local DIS MEDIUM finest0 patch8 stride3 gradient25 variational5; gray source -> cvtColor v12 pulled frame; independently reverse','gate':'cycle<=.5px, norm<=4px, source Sobel magnitude>=4; valid flow sample >=2px; confidence exp(-cycle^2/(2*.5^2)); Gaussian sigma8 normalized mass+.05; 64px smoothstep halo; alpha1','sampling':'exact float bilinear composition, last-cell linear extrapolation; original target color once INTER_LINEAR; no image filtering','fit_independent_of_eval':True,'evaluation':'u fixed NCC; imread0 source, cvtColor all candidate stages; fixed query grid selected by frozen source rectangles+64px halo; no per-eval tuning','limits':{'cpu':2,'gpu':0,'child_seconds':85,'private_mib':900},'v12_generation_ids':[18,209],'v12_source_script_sha256':sha(ROOT/'research/current_chat_flow_boundary_v12_directional_20261024/run.py'),'v13':'only pair208 exists; do not fabricate v13 for other pairs','no_promotion':True,'script_sha256':sha(__file__),'maximum_candidate_runs_per_id':1}
    write(OUT/'POLICY_FROZEN.json',policy);write(HERE/'MULTI_POLICY_FROZEN.json',policy)
    original=(ROOT/'research/current_chat_flow_boundary_v12_directional_20261024/run.py').read_text()
    # Preserve the exact v12 fitting function, omit unrelated original full-frame evaluator and launcher.
    text=original[:original.index(' q=u.query_grid(w,h);')]
    text=text.replace("OUT=Path('C:/Users/thx11/Documents/Codex/NR_models_artifacts/20261024/research_current_chat_flow_boundary_v12_directional')",f'OUT=Path({str(OUT/"v12_generated")!r})')
    text=text.replace("BASE=OUT.parent/'research_current_chat_flow_v5_hardcases'",f'BASE=Path({str(BULK/"research_current_chat_flow_v5_hardcases")!r})')
    text+="\nif __name__=='__main__':work(int(sys.argv[1]))\n"
    (HERE/'multi_v12_fit_adapter.py').write_text(text,encoding='utf-8')
    write(OUT/'V12_ADAPTER_RECEIPT.json',{'original_sha256':policy['v12_source_script_sha256'],'adapter_sha256':sha(HERE/'multi_v12_fit_adapter.py'),'changes':'OUT and BASE path only in fitting body; remove original independent full-frame evaluation and launcher; same v12 fitting math'})

def supervise(stage,ident):
    OUT.mkdir(parents=True,exist_ok=True)
    t=time.monotonic();peak=0;reason=None
    cmd=[sys.executable,'-B',str(HERE/'multi_v12_fit_adapter.py'),str(ident)] if stage=='v12' else [sys.executable,'-B',str(Path(__file__).resolve()),'--child',stage,str(ident)]
    log=OUT/f'{stage}_{ident}.log'
    with log.open('x',encoding='utf-8') as f:
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));ps=psutil.Process(p.pid)
        ps.cpu_affinity(ps.cpu_affinity()[:2])
        while p.poll() is None:
            try:peak=max(peak,ps.memory_info().private/1048576)
            except psutil.Error:pass
            if peak>900 or time.monotonic()-t>85:reason='private900MiB_or_wall85s';p.kill();break
            time.sleep(.1)
        rc=p.wait()
    receipt={'stage':stage,'id':ident,'pid':p.pid,'wall_seconds':time.monotonic()-t,'peak_private_mib_sampled':peak,'sampling_seconds':.1,'exit_code':rc,'termination_reason':reason,'cpu':2,'gpu':0}
    write(OUT/f'{stage}_{ident}_RESOURCE.json',receipt);print(json.dumps(receipt),flush=True)
    if rc:print(log.read_text(encoding='utf-8'),flush=True)

def child(stage,i):
    import cv2,numpy as np,importlib.util
    ps=psutil.Process();ps.cpu_affinity(ps.cpu_affinity()[:2]);cv2.setNumThreads(2);cv2.ocl.setUseOpenCL(False)
    sp=importlib.util.spec_from_file_location('u',ROOT/'research/current_chat_flow_real_v2_detail_20261024/detail_residual_v2.py');u=importlib.util.module_from_spec(sp);sp.loader.exec_module(u)
    dst=OUT/('review_208' if stage=='review208' else f'Train_{i}');dst.mkdir(parents=True,exist_ok=False)
    source_path=ROOT/f'Train/input_f/{i}.png';target_path=ROOT/f'Train/output_f/{i}.png'
    source=cv2.imread(str(source_path));sg=cv2.imread(str(source_path),0);target=cv2.imread(str(target_path));h,w=sg.shape
    def panel(im,label,box):
        a,b,c,d=map(int,box);crop=im[b:d,a:c];head=np.zeros((28,c-a,3),np.uint8);cv2.putText(head,label,(4,20),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1,cv2.LINE_AA);return np.vstack((head,crop))
    def tiles(box):
        a,b,c,d=box
        return [[x,y,min(x+360,c),min(y+400,d)] for y in range(b,d,400) for x in range(a,c,360)]
    def overlays(ims,box,prefix):
        names=[]
        for t,b in enumerate(tiles(box)):
            parts=[panel(cv2.addWeighted(source,.5,im,.5,0),f'SRC + {name.upper()} 50%',b) for name,im in ims.items()]
            if len(parts)>3:
                # For four stages show two paired files; every stage stays native scale.
                for offset in (0,2):
                    name=f'{prefix}_tile{t:02d}_group{offset//2}_overlay_native.png';cv2.imwrite(str(dst/name),np.hstack(parts[offset:offset+2]));names.append(name)
            else:
                name=f'{prefix}_tile{t:02d}_overlay_native.png';cv2.imwrite(str(dst/name),np.hstack(parts));names.append(name)
        return names
    if stage=='review208':
        paths={'v5':v5(208)/'aligned_target_on_source.png','v13':BULK/'research_current_chat_foot_local_v13/Train_208/aligned_target_on_source.png','v14':BULK/'research_current_chat_foot_local_v14/Train_208/aligned_target_on_source.png'}
        ims={k:cv2.imread(str(p)) for k,p in paths.items()};views=[]
        points=[[944,976],[944,944],[912,976],[816,944]]
        z=np.load(BULK/'research_current_chat_foot_local_v14/Train_208/INDEPENDENT_EVAL.npz');diagnostics=[]
        for x,y in points:
            views+=overlays(ims,[x-64,y-64,x+64,y+64],f'x{x}_y{y}')
            idx=np.where(np.all(z['query_xy']==[x,y],axis=1))[0][0]
            diagnostics.append({'xy':[x,y],**{k:{'known':bool(z[k+'_known'][idx]),'NCC_px':float(z[k+'_error'][idx])} for k in ('v12','v13','v14')}})
        views+=overlays(ims,[840,1032,1192,1248],'shoes')
        write(dst/'REVIEW_INPUTS.json',{'points':diagnostics,'views':views,'hashes':{k:sha(p) for k,p in paths.items()},'role':'posthoc requested diagnostics only; no fit, no fresh validation claim','user_v13_accepted_softness':True,'contour_geometry_acceptance':'UNKNOWN_pending_view'})
        return
    boxes=rows()[i];bp=v12(i);old=v5(i)
    inputs={'source':source_path,'target':target_path,'v12_map':bp/'composite_source_to_target_float32.npy','v12_png':bp/'aligned_target_on_source.png','v5_map':old/'composite_source_to_target_float32.npy','v5_png':old/'aligned_target_on_source.png','annotation':ROOT/'coordination/v5_hardcase_fold_locations_20261024.json','evaluator':u.EVAL,'u':Path(u.__file__)}
    hashes={k:sha(p) for k,p in inputs.items()}
    parent=cv2.imread(str(inputs['v12_png']));pg=cv2.cvtColor(parent,cv2.COLOR_BGR2GRAY);oldim=cv2.imread(str(inputs['v5_png']))
    m=np.load(inputs['v12_map']);new=m.copy();flowinfo=[]
    changed_domain=np.zeros((h,w),bool)
    for j,box in enumerate(boxes):
        a,b,c,d=box;x0=max(0,a-64);y0=max(0,b-64);x1=min(w,c+64);y1=min(h,d+64)
        s=sg[y0:y1,x0:x1].copy();p=pg[y0:y1,x0:x1].copy()
        def flow(a,b):
            dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM);dis.setFinestScale(0);dis.setPatchSize(8);dis.setPatchStride(3);dis.setGradientDescentIterations(25);dis.setVariationalRefinementIterations(5);return dis.calc(a,b,None)
        f=flow(s,p);r=flow(p,s);yy,xx=np.mgrid[:s.shape[0],:s.shape[1]].astype(np.float32)
        rx=xx+f[:,:,0];ry=yy+f[:,:,1];br=cv2.remap(r,rx,ry,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        cycle=np.linalg.norm(f+br,axis=2);length=np.linalg.norm(f,axis=2)
        grad=cv2.magnitude(cv2.Sobel(s,cv2.CV_32F,1,0),cv2.Sobel(s,cv2.CV_32F,0,1))
        known=(cycle<=.5)&(length<=4)&(grad>=4)&(rx>=2)&(rx<s.shape[1]-2)&(ry>=2)&(ry<s.shape[0]-2)
        conf=known.astype(np.float32)*np.exp(-cycle**2/(2*.5**2));mass=cv2.GaussianBlur(conf,(0,0),8);num=cv2.GaussianBlur(f*conf[:,:,None],(0,0),8);field=num/(mass[:,:,None]+.05)
        X=xx+x0;Y=yy+y0;dist=np.sqrt(np.maximum(np.maximum(a-X,X-(c-1)),0)**2+np.maximum(np.maximum(b-Y,Y-(d-1)),0)**2)
        gate=np.clip(1-dist/64,0,1);gate=gate*gate*(3-2*gate);field*=gate[:,:,None]
        sx=X+field[:,:,0];sy=Y+field[:,:,1];bx=np.clip(np.floor(sx).astype(np.int32),0,w-2);by=np.clip(np.floor(sy).astype(np.int32),0,h-2);fx=(sx-bx)[:,:,None];fy=(sy-by)[:,:,None]
        comp=((1-fy)*((1-fx)*m[by,bx]+fx*m[by,bx+1])+fy*((1-fx)*m[by+1,bx]+fx*m[by+1,bx+1])).astype(np.float32)
        active=np.any(field!=0,axis=2)
        assert not (changed_domain[y0:y1,x0:x1]&active).any(),'Overlapping independently applied corrections not frozen'
        new[y0:y1,x0:x1][active]=comp[active];changed_domain[y0:y1,x0:x1]|=active
        np.savez_compressed(dst/f'person{j}_DIS.npz',source_roi=np.array(box),halo=np.array([x0,y0,x1,y1]),forward=f,reverse=r,known=known,cycle_px=cycle,field=field)
        flowinfo.append({'person':j,'box':box,'halo':[x0,y0,x1,y1],'solver_known':int(known.sum()),'solver_UNKNOWN':int((~known).sum()),'field_max_px':float(np.linalg.norm(field,axis=2).max())})
    np.save(dst/'composite_source_to_target_float32.npy',new)
    cand=cv2.remap(target,new[:,:,0],new[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);cv2.imwrite(str(dst/'aligned_target_on_source.png'),cand)
    th,tw=target.shape[:2]
    def valid(a):return np.isfinite(a).all(2)&(a[:,:,0]>=0)&(a[:,:,0]<=tw-1)&(a[:,:,1]>=0)&(a[:,:,1]<=th-1)
    sb=valid(m);sn=valid(new);jac=u.jacobian_min(new);geom={'v12_minJac':u.jacobian_min(m),'v14_minJac':jac,'added_support_loss_vs_v12':int((sb&~sn).sum()),'v12_invalid_support':int((~sb).sum()),'v14_invalid_support':int((~sn).sum()),'outside_local_domain_changed':int((np.any(new!=m,axis=2)&~changed_domain).sum())};geom['pass']=jac>=.05 and geom['added_support_loss_vs_v12']==0
    # Freeze exact candidate hashes before invoking the independent evaluator.
    write(dst/'CANDIDATE_FROZEN_BEFORE_EVAL.json',{'map_sha256':sha(dst/'composite_source_to_target_float32.npy'),'png_sha256':sha(dst/'aligned_target_on_source.png'),'policy_sha256':sha(OUT/'POLICY_FROZEN.json'),'geometry':geom,'new_solver_observations':flowinfo})
    q0=u.query_grid(w,h);sel=np.zeros(len(q0),bool)
    for a,b,c,d in boxes:sel|=(q0[:,0]>=a-64)&(q0[:,0]<c+64)&(q0[:,1]>=b-64)&(q0[:,1]<d+64)
    q=q0[sel];stages={};records={'query_xy':q}
    for label,im in [('v5',oldim),('v12',parent),('v14',cand)]:
        k,e=u.independent_eval(sg,cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),q);stages[label]=(k,e);records[label+'_known']=k;records[label+'_error']=e
    np.savez_compressed(dst/'INDEPENDENT_EVAL.npz',**records)
    def compare(before,mask):
        kb,eb=stages[before];ka,ea=stages['v14'];co=kb&ka&mask;ids=np.where(co)[0];worst=ids[np.argsort(ea[ids]-eb[ids])[::-1]][:8]
        return {'denominator':int(mask.sum()),'before_known':int((kb&mask).sum()),'v14_known':int((ka&mask).sum()),'before_UNKNOWN':int((~kb&mask).sum()),'v14_UNKNOWN':int((~ka&mask).sum()),'lost':int((kb&~ka&mask).sum()),'gained':int((~kb&ka&mask).sum()),'common':int(co.sum()),'before_common':u.stat(eb[co]),'v14_common':u.stat(ea[co]),'worst_regressions':[{'xy':q[t].tolist(),'before_px':float(eb[t]),'v14_px':float(ea[t]),'regression_px':float(ea[t]-eb[t])} for t in worst]}
    masks={'local_plus_halo':np.ones(len(q),bool)}
    for j,(a,b,c,d) in enumerate(boxes):masks[f'person{j}']=(q[:,0]>=a)&(q[:,0]<c)&(q[:,1]>=b)&(q[:,1]<d)
    evaluation={f'{b}_to_v14':{name:compare(b,mask) for name,mask in masks.items()} for b in ('v5','v12')}
    write(dst/'INDEPENDENT_EVAL.json',evaluation)
    views=[]
    for j,box in enumerate(boxes):
        views+=overlays({'v5':oldim,'v14':cand},box,f'person{j}_v5_v14')
        views+=overlays({'v12':parent,'v14':cand},box,f'person{j}_v12_v14')
        for t,bb in enumerate(tiles(box)):
            # Extra unfiltered RGB and red/cyan diagnostics, native crops.
            cv2.imwrite(str(dst/f'person{j}_tile{t:02d}_RGB_native.png'),np.hstack([panel(im,label,bb) for label,im in [('SOURCE',source),('V5',oldim),('V14',cand)]]))
            diag=np.stack([cv2.cvtColor(cand,cv2.COLOR_BGR2GRAY),cv2.cvtColor(cand,cv2.COLOR_BGR2GRAY),sg],axis=2)
            cv2.imwrite(str(dst/f'person{j}_tile{t:02d}_RED_CYAN_native.png'),panel(diag,'SOURCE RED / V14 CYAN',bb))
    # Save maximum observed regression patches as posthoc diagnostics, never refit.
    for before in ('v5','v12'):
        worst=evaluation[f'{before}_to_v14']['local_plus_halo']['worst_regressions']
        if worst:
            x,y=map(int,worst[0]['xy']);box=[max(0,x-80),max(0,y-80),min(w,x+80),min(h,y+80)]
            views+=overlays({before:oldim if before=='v5' else parent,'v14':cand},box,f'worst_vs_{before}')
    hashes_after={k:sha(p) for k,p in inputs.items()}
    result={'id':i,'geometry':geom,'input_hashes':hashes,'inputs_unchanged':hashes==hashes_after,'regions':boxes,'new_solver_observations':flowinfo,'views':views,'v13':'NOT_AVAILABLE_NOT_FABRICATED','character_edges_lt1_5_proven':False,'visual_review':'PENDING','promotion':False,'not_GT':True,'eval_points_used_to_fit':False,'sample_once_replay_different_channels':int(np.count_nonzero(cand!=cv2.remap(target,new[:,:,0],new[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)))}
    write(dst/'RESULT.json',result)
    print(json.dumps({'id':i,'geometry':geom,'evaluation':{k:v['local_plus_halo'] for k,v in evaluation.items()}},indent=2),flush=True)

if __name__=='__main__':
    args=sys.argv[1:]
    if args==['--freeze']:setup()
    elif args and args[0]=='--child':child(args[1],int(args[2]))
    else:supervise(args[0],int(args[1]))
