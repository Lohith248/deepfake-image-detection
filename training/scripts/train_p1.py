"""Versioned P1 confirmation and P2 loss screen on the frozen pool only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import base64
import copy
import csv
import gc
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from deepfake_core import build_model,decode_rgb,to_tensor,sha256,save_bundle
from e1_transforms import conditions,transform,stable_seed,training_view
from e1_metrics import auc,sigmoid64,summarize


def emit_artifact(name,data):
    if len(data)>8*1024*1024 or Path(name).name!=name: raise ValueError('Invalid artifact')
    print('P1_ARTIFACT '+name+' '+base64.b64encode(data).decode(),flush=True)


def emit_json(name,value):
    emit_artifact(name,(json.dumps(value,indent=2)+'\n').encode())


def readiness(check_free=True):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
    uuid,free=[v.strip() for v in rows[0].split(',')]
    if uuid!='GPU-4f7dd3a1-55b5-1d57-d0ad-00186da5363e' or (check_free and float(free)<10000):raise RuntimeError('Primary GPU unavailable')
    listing=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    others=[line.strip() for line in listing.splitlines() if line.strip().isdigit() and int(line)!=os.getpid()]
    if others:raise RuntimeError('Other compute process is using the GPU; no competing job started')


class FitSet(Dataset):
    def __init__(self,arrays,rows,indices,fold,seed):self.arrays=arrays;self.rows=rows;self.indices=indices;self.fold=fold;self.seed=seed;self.epoch=0
    def __len__(self):return len(self.indices)
    def __getitem__(self,n):
        i=self.indices[n];r=self.rows[i]
        image=training_view(self.arrays[i],r['image_id'],self.epoch,self.seed,self.fold)
        return to_tensor(image),float(r['label'])


@torch.inference_mode()
def logits_for(model,images,batch_size):
    model.eval();result=[]
    for start in range(0,len(images),batch_size):
        x=torch.stack([to_tensor(a) for a in images[start:start+batch_size]]).to('cuda')
        z=model(x).reshape(-1).float()
        if not torch.isfinite(z).all():raise ValueError('Non-finite validation logits')
        result.extend(z.cpu().tolist())
    return np.asarray(result,dtype=np.float64)


def build_validation_cache(arrays,rows,indices,cs,workers):
    # About9.5GB at1191images/54conditions, RAM-only, reused by all4models.
    cache=np.empty((len(cs),len(indices),224,224,3),dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for ci,c in enumerate(cs):
            def generate(pair):
                vi,i=pair;return vi,transform(arrays[i],c['operator'],c['severity'],stable_seed('eval:20261031:'+rows[i]['image_id']+':'+c['name']))
            for vi,image in executor.map(generate,enumerate(indices)):cache[ci,vi]=image
            if ci%10==0:print('P1_CACHE '+json.dumps({'condition':ci,'of':len(cs),'name':c['name']}),flush=True)
    return cache


def optimizer_for(model,architecture,config):
    groups={}
    for name,p in model.named_parameters():
        head=name.startswith('classifier.1.') if architecture.startswith('efficientnet') else name.startswith('classifier.2.') if architecture=='convnext_tiny' else name.startswith('head.')
        decay=p.ndim>1 and not name.endswith('.bias');key=(head,decay)
        groups.setdefault(key,[]).append(p)
    return torch.optim.AdamW([{'params':params,'lr':config['head_lr'] if head else config['backbone_lr'],
        'base_lr':config['head_lr'] if head else config['backbone_lr'],'weight_decay':config['weight_decay'] if decay else 0.}
        for (head,decay),params in groups.items()])



MANIFEST_HASH='30f7bb575793da6ab0b119abdaa96b8377bc44645de134035517a09bc0f75c85'
WEIGHTS={'convnext_tiny':('convnext_tiny-983f1562.pth','983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d'),
         'swin_t':('swin_t-704ceda3.pth','704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b')}
EXPECTED=[('p1',f,a,17,'bce') for f in (2,3,4) for a in ('convnext_tiny','swin_t')]+[('p1',f,'convnext_tiny',29,'bce') for f in range(5)]+[('p2',f,'convnext_tiny',17,l) for l in ('gce_q07','smooth_01') for f in (0,1)]


def fit_id(job):
    return f"{job['stage']}_f{job['fold']}_{job['architecture']}_s{job['seed']}_{job['loss']}"


def check_protocol(protocol):
    actual=[(j['stage'],j['fold'],j['architecture'],j['seed'],j['loss']) for j in protocol['jobs']]
    if actual!=EXPECTED or protocol['evaluation_epochs']!=[20]: raise ValueError('Protocol schedule changed')
    if protocol['e1_protocol_sha256']!='ad0f1f75abac558dca4a2e1459463b8ce71200ad7b56c0bb7acb82cbfad8d1e8': raise ValueError('E1 protocol identity changed')


def training_loss(z,y,kind):
    if kind=='bce': return torch.nn.functional.binary_cross_entropy_with_logits(z,y)
    if kind=='smooth_01': return torch.nn.functional.binary_cross_entropy_with_logits(z,.9*y+.05)
    if kind=='gce_q07': return (-torch.expm1(.7*torch.nn.functional.logsigmoid((2*y.float()-1)*z.float()))/.7).mean()
    raise ValueError('Unknown loss')


def pool_path(root,relative):
    base=(root/'pool-images').resolve(strict=True)
    if not base.is_relative_to(root.resolve()) or (root/'pool-images').is_symlink(): raise ValueError('Unsafe pool root')
    path=(root/relative).resolve(strict=True)
    if not path.is_relative_to(base) or path==base: raise ValueError('Image outside pool')
    return path


def verify_pool(root):
    manifest=root/'pool_manifest.csv'
    if sha256(manifest)!=MANIFEST_HASH: raise ValueError('Pool manifest changed')
    with manifest.open(newline='') as f: rows=list(csv.DictReader(f))
    if len(rows)!=5950 or any(r['partition']!='pool' for r in rows) or len({r['image_id'] for r in rows})!=len(rows): raise ValueError('Invalid pool membership')
    groups={}
    for r in rows:
        if r['fold'] not in ('0','1','2','3','4') or r['label'] not in ('0','1'): raise ValueError('Invalid pool row')
        if r['group_id'] in groups and groups[r['group_id']]!=r['fold']: raise ValueError('Source group crosses folds')
        groups[r['group_id']]=r['fold']
        if sha256(pool_path(root,r['relative_path']))!=r['sha256']: raise ValueError('Pool image hash mismatch')
    return rows


def source_identity(root):
    listing=root/'source-hashes.json'; hashes=json.loads(listing.read_text())
    required={'deepfake_core.py','e1_transforms.py','e1_metrics.py','scripts/train_e1.py','scripts/train_p1.py','inference_script.py','experiments/e1_protocol.json','experiments/p1_protocol.json'}
    if not required.issubset(hashes): raise ValueError('Missing source identity')
    for name,digest in hashes.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or sha256(path)!=digest: raise ValueError('Source changed: '+name)
    if sha256(root/'experiments/e1_protocol.json')!='ad0f1f75abac558dca4a2e1459463b8ce71200ad7b56c0bb7acb82cbfad8d1e8': raise ValueError('E1 protocol changed')
    return {'source_hashes_sha256':sha256(listing),'source_hashes':hashes}


def verify_weights(cache):
    for name,(filename,digest) in WEIGHTS.items():
        path=cache/filename
        if not path.is_file() or sha256(path)!=digest: raise ValueError('Missing or changed ImageNet weights: '+name)


def validate_completed(directory,identity):
    receipt=directory/'receipt.json'
    if not receipt.is_file(): raise ValueError('Incomplete final fit')
    record=json.loads(receipt.read_text())
    if record['identity']!=identity: raise ValueError('Completed fit identity changed')
    files=record['artifacts']
    if set(files)!={'checkpoint.pt','predictions.npz','curves.json','summary.json','cli.csv'}: raise ValueError('Invalid completion receipt')
    if {p.name for p in directory.iterdir()}!=set(files)|{'receipt.json'}: raise ValueError('Unexpected completed fit contents')
    for name,digest in files.items():
        if sha256(directory/name)!=digest: raise ValueError('Completed artifact changed: '+name)
    summary=json.loads((directory/'summary.json').read_text())
    if summary.get('fit_id')!=directory.name or fit_id(summary)!=directory.name or summary.get('identity')!=identity:
        raise ValueError('Completed fit summary belongs to another fit')
    return summary


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def publish_artifacts(directory,prefix):
    for source,suffix in [('summary.json','summary.json'),('curves.json','curves.json'),('predictions.npz','e20_predictions.npz')]:
        emit_artifact(prefix+'_'+suffix,(directory/source).read_bytes())


def emit_gate(out,name,jobs,identity):
    record={'fits':[fit_id(j) for j in jobs], 'identity':identity,
            'receipts':{fit_id(j):sha256(out/fit_id(j)/'receipt.json') for j in jobs}}
    target=out/name
    if target.exists():
        if json.loads(target.read_text())!=record: raise ValueError('Completion gate changed: '+name)
    else:
        temp=out/('.'+name+'.'+str(os.getpid())+'.tmp');write_json(temp,record)
        with temp.open('rb') as f: os.fsync(f.fileno())
        os.replace(temp,target)
        fd=os.open(out,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    emit_json(name,record)


def run_experiment(root,stage='all'):
    import tempfile, uuid
    root=root.resolve();protocol=json.loads((root/'experiments/p1_protocol.json').read_text());check_protocol(protocol)
    identity=source_identity(root); rows=verify_pool(root)
    cache_weights=Path.home()/'deepfake_hackathon/e0-20261031-v1/weights';verify_weights(cache_weights)
    if sys.version_info[:2]!=(3,10) or not torch.cuda.is_available(): raise RuntimeError('Remote Python3.10/CUDA required')
    if '5060 Ti' not in torch.cuda.get_device_name(0): raise RuntimeError('Refuse laptop/other GPU')
    readiness();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.75)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True,warn_only=True)
    import torchvision,PIL
    cs=conditions(); identity.update({'pool_manifest_sha256':MANIFEST_HASH,'weights':WEIGHTS,'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'pillow':PIL.__version__,'gpu':torch.cuda.get_device_name(0),'gpu_uuid':'GPU-4f7dd3a1-55b5-1d57-d0ad-00186da5363e','cuda':torch.version.cuda,'conditions':cs})
    identity=json.loads(json.dumps(identity))
    out=root/'results';out.mkdir(exist_ok=True)
    environment_path=out/'environment.json'
    if environment_path.exists():
        if json.loads(environment_path.read_text())!=identity: raise ValueError('Experiment environment changed')
    else:
        temp=out/'.environment.json.part';write_json(temp,identity);os.replace(temp,environment_path)
    emit_artifact('environment.json',environment_path.read_bytes())
    jobs=protocol['jobs'];completed={}
    for job in jobs:
        final=out/fit_id(job)
        if final.exists():
            completed[fit_id(job)]=validate_completed(final,identity);publish_artifacts(final,fit_id(job))
    if stage=='p2' and any(fit_id(j) not in completed for j in jobs if j['stage']=='p1'): raise ValueError('P1 must complete before P2')
    arrays=None;cache=None;cache_fold=None;labels=np.array([int(r['label']) for r in rows]);start=time.perf_counter();config=protocol['recipe']
    for job in jobs:
        prefix=fit_id(job); final=out/prefix
        if prefix in completed or (stage!='all' and job['stage']!=stage): continue
        if job['stage']=='p2' and any(fit_id(j) not in completed for j in jobs if j['stage']=='p1'): raise ValueError('P1 must complete before P2')
        if job['stage']=='p2': emit_gate(out,'p1_complete.json',[j for j in jobs if j['stage']=='p1'],identity)
        if arrays is None: arrays=np.stack([decode_rgb(pool_path(root,r['relative_path'])) for r in rows])
        fold=job['fold'];architecture=job['architecture'];seed=job['seed']
        val_ix=np.flatnonzero(np.array([int(r['fold']) for r in rows])==fold);fit_ix=np.flatnonzero(np.array([int(r['fold']) for r in rows])!=fold)
        if {rows[i]['group_id'] for i in fit_ix}&{rows[i]['group_id'] for i in val_ix}: raise ValueError('Group leakage')
        if cache_fold!=fold:
            if cache is not None: del cache;gc.collect()
            cache=build_validation_cache(arrays,rows,val_ix,cs,8);cache_fold=fold
        fit_rng=np.random.default_rng(91+fold);fit_sample=[]
        for label in [0,1]:
            candidates=fit_ix[labels[fit_ix]==label];fit_sample.extend(fit_rng.choice(candidates,min(256,len(candidates)),replace=False).tolist())
        readiness();verify_weights(cache_weights);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);run_start=time.perf_counter()
        staging=out/('.attempt-'+prefix+'-'+uuid.uuid4().hex);staging.mkdir()
        def reject_download(*args,**kwargs): raise RuntimeError('Network weight downloads are forbidden')
        original_download=torch.hub.download_url_to_file;torch.hub.download_url_to_file=reject_download
        try: model,provenance=build_model(architecture,pretrained=True,weights_dir=cache_weights)
        finally: torch.hub.download_url_to_file=original_download
        if provenance['sha256']!=WEIGHTS[architecture][1]: raise ValueError('Loaded weights changed')
        model=model.cuda();ema=copy.deepcopy(model).eval()
        for q in ema.parameters():q.requires_grad_(False)
        optimizer=optimizer_for(model,architecture,config);scaler=torch.amp.GradScaler('cuda',init_scale=128.)
        dataset=FitSet(arrays,rows,fit_ix,fold,seed);curves=[];global_step=0;train_seconds=0.;eval_seconds=0.
        print('P1_RUN_START '+json.dumps(job),flush=True)
        for epoch in range(config['epochs']):
            readiness(check_free=False)
            if time.perf_counter()-start>protocol['max_wall_seconds']:raise TimeoutError('P1 cap exceeded')
            tick=time.perf_counter();dataset.epoch=epoch;model.train()
            generator=torch.Generator().manual_seed(seed+fold*1000+epoch)
            loader=DataLoader(dataset,batch_size=config['train_batch_size'],shuffle=True,num_workers=0,pin_memory=True,generator=generator)
            epoch_loss=0.;seen=0;skipped=0
            for batch,(x,y) in enumerate(loader):
                progress=epoch+(batch+1)/len(loader)
                factor=.1+.9*progress/config['warmup_epochs'] if progress<=config['warmup_epochs'] else .5*(1+math.cos(math.pi*(progress-config['warmup_epochs'])/(config['epochs']-config['warmup_epochs'])))
                for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
                x=x.cuda(non_blocking=True);y=y.float().cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    z=model(x).reshape(-1);loss=training_loss(z,y,job['loss'])
                if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
                scale_before=scaler.get_scale();scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
                if scaler.get_scale()<scale_before:skipped+=1
                else:
                    with torch.no_grad():
                        for q,p0 in zip(ema.parameters(),model.parameters()):q.mul_(config['ema_decay']).add_(p0,alpha=1-config['ema_decay'])
                        for q,b0 in zip(ema.buffers(),model.buffers()):q.copy_(b0)
                epoch_loss+=float(loss.detach())*len(y);seen+=len(y);global_step+=1
            torch.cuda.synchronize();seconds=time.perf_counter()-tick;train_seconds+=seconds
            if skipped>max(2,.05*len(loader)):raise ValueError('Too many non-finite AMP steps')
            row={'epoch':epoch+1,'loss':epoch_loss/seen,'train_seconds':seconds,'optimizer_steps':global_step,'amp_skipped_steps':skipped,'last_lr':optimizer.param_groups[0]['lr']}
            if epoch+1==config['epochs']:
                eval_start=time.perf_counter();scores=np.empty((len(cs),len(val_ix)),dtype=np.float64);logits=np.empty_like(scores)
                for ci in range(len(cs)):
                    logits[ci]=logits_for(ema,cache[ci],config['eval_batch_size']);scores[ci]=sigmoid64(logits[ci])
                metrics=summarize(labels[val_ix],scores,cs);raw=sigmoid64(logits_for(model,cache[0],config['eval_batch_size']))
                fit_scores=sigmoid64(logits_for(ema,arrays[fit_sample],config['eval_batch_size']))
                row.update({'metrics':metrics,'raw_original_auc':auc(labels[val_ix],raw),'fit_sample_auc':auc(labels[fit_sample],fit_scores)})
                eval_seconds+=time.perf_counter()-eval_start
                np.savez_compressed(staging/'predictions.npz',probabilities=scores,logits=logits,raw_original_probabilities=raw,y=labels[val_ix],image_ids=np.array([rows[i]['image_id'] for i in val_ix]),group_ids=np.array([rows[i]['group_id'] for i in val_ix]),condition_names=np.array([c['name'] for c in cs]),condition_weights=np.array([c['weight'] for c in cs]))
                print('P1_METRICS '+json.dumps({'fit_id':prefix,**metrics}),flush=True)
            curves.append(row);print('P1_EPOCH '+json.dumps({'fit_id':prefix,**{k:v for k,v in row.items() if k!='metrics'}}),flush=True)
        checkpoint=staging/'checkpoint.pt';write_json(staging/'curves.json',curves)
        save_bundle(checkpoint,architecture,ema,provenance,purpose='P1 development checkpoint; not final holdout evaluated',training_metadata={**job,'epochs':config['epochs'],'protocol_sha256':identity['source_hashes']['experiments/p1_protocol.json']})
        with tempfile.TemporaryDirectory(prefix='p1-cli-') as tmp:
            fixture=Path(tmp)
            for i in val_ix[:16]: (fixture/(rows[i]['image_id']+'.jpg')).symlink_to(pool_path(root,rows[i]['relative_path']))
            cli_output=staging/'cli.csv';command=[sys.executable,str(root/'inference_script.py'),'--test_dir',str(fixture),'--weights',str(checkpoint),'--output',str(cli_output),'--batch_size','64']
            subprocess.run(command,check=True,timeout=120);first_csv=cli_output.read_bytes();subprocess.run(command,check=True,timeout=120)
            if first_csv!=cli_output.read_bytes():raise ValueError('CLI replay differs')
            with cli_output.open(newline='') as f:cli_rows=list(csv.DictReader(f))
            mapped={r['image_id']:float(r['confidence']) for r in cli_rows}
            if len(cli_rows)!=16 or set(mapped)!={rows[i]['image_id'] for i in val_ix[:16]}:raise ValueError('CLI ID mapping changed')
            error=max(abs(mapped[rows[i]['image_id']]-scores[0,j]) for j,i in enumerate(val_ix[:16]))
            if error>1e-5:raise ValueError('CLI/validation score mismatch')
        summary={**job,'fit_id':prefix,'epochs':config['epochs'],'fit_images':len(fit_ix),'validation_images':len(val_ix),'train_seconds':train_seconds,'evaluation_seconds':eval_seconds,'wall_seconds':time.perf_counter()-run_start,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'final_metrics':curves[-1]['metrics'],'checkpoint_sha256':sha256(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,'provenance':provenance,'cli_replay_byte_identical':True,'cli_validation_max_probability_difference':error,'identity':identity}
        write_json(staging/'summary.json',summary)
        artifacts={p.name:sha256(p) for p in staging.iterdir()};write_json(staging/'receipt.json',{'identity':identity,'artifacts':artifacts})
        for p in staging.iterdir():
            with p.open('rb') as f:os.fsync(f.fileno())
        if final.exists():raise ValueError('Final fit already exists')
        staging_fd=os.open(staging,os.O_RDONLY)
        try: os.fsync(staging_fd)
        finally: os.close(staging_fd)
        staging.rename(final)
        out_fd=os.open(out,os.O_RDONLY)
        try: os.fsync(out_fd)
        finally: os.close(out_fd)
        completed[prefix]=validate_completed(final,identity);publish_artifacts(final,prefix)
        del optimizer,scaler,model,ema,x,y,z,loss;gc.collect();torch.cuda.empty_cache()
    for part in ('p1','p2'):
        selected=[j for j in jobs if j['stage']==part]
        if all(fit_id(j) in completed for j in selected): emit_gate(out,part+'_complete.json',selected,identity)
    if len(completed)==len(jobs):emit_gate(out,'complete.json',jobs,identity)


def main():
    import fcntl
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--stage',choices=('p1','p2','all'),default='all');args=p.parse_args()
    lock=args.root.resolve()/'.p1.lock'
    with lock.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another P1 process holds the lock')
        run_experiment(args.root,args.stage)


if __name__=='__main__':main()
