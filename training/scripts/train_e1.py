"""Matched E1 pool-only screen. Never opens final holdout/test images."""
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
    print('E1_ARTIFACT '+name+' '+base64.b64encode(data).decode(),flush=True)


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
            if ci%10==0:print('E1_CACHE '+json.dumps({'condition':ci,'of':len(cs),'name':c['name']}),flush=True)
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


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=p.parse_args()
    root=args.root.resolve();config=json.loads((root/'experiments/e1_protocol.json').read_text());start=time.perf_counter()
    if sys.version_info[:2]!=(3,10) or not torch.cuda.is_available():raise RuntimeError('Remote Python3.10/CUDA required')
    if '5060 Ti' not in torch.cuda.get_device_name(0):raise RuntimeError('Refuse laptop/other GPU')
    readiness();torch.set_num_threads(8);torch.cuda.set_per_process_memory_fraction(.75)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    with (root/'pool_manifest.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=5950 or any(r['partition']!='pool' for r in rows):raise ValueError('Only5950pool images are allowed')
    if len({r['image_id'] for r in rows})!=len(rows):raise ValueError('Duplicate pool IDs')
    groups={}
    for r in rows:
        if r['group_id'] in groups and groups[r['group_id']]!=r['fold']:raise ValueError('Source group crosses folds')
        groups[r['group_id']]=r['fold']
        if sha256(root/r['relative_path'])!=r['sha256']:raise ValueError('Pool image hash mismatch')
    arrays=np.stack([decode_rgb(root/r['relative_path']) for r in rows]);labels=np.array([int(r['label']) for r in rows])
    cs=conditions();out=root/'results';out.mkdir(exist_ok=True)
    import torchvision,PIL
    environment={'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'pillow':PIL.__version__,
        'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'tf32':False,'cudnn_deterministic':True,'seed':config['seed'],
        'pool_manifest_sha256':sha256(root/'pool_manifest.csv'),'protocol_sha256':sha256(root/'experiments/e1_protocol.json'),
        'source_hashes':{name:sha256(root/name) for name in ['deepfake_core.py','e1_transforms.py','e1_metrics.py','scripts/train_e1.py','inference_script.py']},
        'fit_scope':'pool fold complement only; fresh ImageNet initialization; no E0 trained weights',
        'conditions':cs,'training_view':'A1-light; convolution/geometric excluded from extra augmentation only'}
    emit_json('environment.json',environment);completed=[]
    weights_cache=Path.home()/'deepfake_hackathon/e0-20261031-v1/weights'
    for fold in config['folds']:
        val_ix=np.flatnonzero(np.array([int(r['fold']) for r in rows])==fold);fit_ix=np.flatnonzero(np.array([int(r['fold']) for r in rows])!=fold)
        assert not {rows[i]['group_id'] for i in fit_ix}&{rows[i]['group_id'] for i in val_ix}
        cache_start=time.perf_counter();cache=build_validation_cache(arrays,rows,val_ix,cs,8)
        print('E1_CACHE_READY '+json.dumps({'fold':fold,'bytes':cache.nbytes,'seconds':time.perf_counter()-cache_start}),flush=True)
        fit_rng=np.random.default_rng(91+fold);fit_sample=[]
        for label in [0,1]:
            candidates=fit_ix[labels[fit_ix]==label];fit_sample.extend(fit_rng.choice(candidates,min(256,len(candidates)),replace=False).tolist())
        for architecture in config['architectures']:
            if time.perf_counter()-start>config['max_wall_seconds']:raise TimeoutError('E1 cap exceeded')
            readiness();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            seed=config['seed'];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            prefix=f'f{fold}_{architecture}';run_start=time.perf_counter()
            summary_path=out/(prefix+'_summary.json')
            if summary_path.exists():
                old=json.loads(summary_path.read_text())
                if old['source_hashes']!=environment['source_hashes'] or old['protocol_sha256']!=environment['protocol_sha256'] or old['pool_manifest_sha256']!=environment['pool_manifest_sha256']:
                    raise ValueError('Existing run belongs to different inputs/code; use a new experiment version')
                if sha256(Path(old['checkpoint_remote_path']))!=old['checkpoint_sha256']:raise ValueError('Existing checkpoint changed')
                for filename in [prefix+'_summary.json',prefix+'_curves.json',prefix+'_e10_predictions.npz',prefix+'_e20_predictions.npz']:
                    emit_artifact(filename,(out/filename).read_bytes())
                completed.append(old);print('E1_RESUMED_COMPLETED '+prefix,flush=True);continue
            model,provenance=build_model(architecture,pretrained=True,weights_dir=weights_cache)
            model=model.cuda();ema=copy.deepcopy(model).eval()
            for q in ema.parameters():q.requires_grad_(False)
            optimizer=optimizer_for(model,architecture,config);scaler=torch.amp.GradScaler('cuda',init_scale=128.)
            dataset=FitSet(arrays,rows,fit_ix,fold,seed)
            curves=[];global_step=0;train_seconds=0.;eval_seconds=0.
            print('E1_RUN_START '+json.dumps({'fold':fold,'architecture':architecture,'fit':len(fit_ix),'validation':len(val_ix)}),flush=True)
            for epoch in range(config['epochs']):
                readiness(check_free=False)
                if time.perf_counter()-start>config['max_wall_seconds']:raise TimeoutError('E1 cap exceeded')
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
                        z=model(x).reshape(-1);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y)
                    if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
                    scale_before=scaler.get_scale();scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
                    if scaler.get_scale()<scale_before:skipped+=1
                    else:
                        with torch.no_grad():
                            for q,p0 in zip(ema.parameters(),model.parameters()):q.mul_(config['ema_decay']).add_(p0,alpha=1-config['ema_decay'])
                            for q,b0 in zip(ema.buffers(),model.buffers()):q.copy_(b0)
                    epoch_loss+=float(loss.detach())*len(y);seen+=len(y);global_step+=1
                torch.cuda.synchronize();seconds=time.perf_counter()-tick;train_seconds+=seconds
                if skipped>max(2,.05*len(loader)):raise ValueError('Too many non-finite AMP steps; diagnose before comparing models')
                row={'epoch':epoch+1,'loss':epoch_loss/seen,'train_seconds':seconds,'optimizer_steps':global_step,'amp_skipped_steps':skipped,'last_lr':optimizer.param_groups[0]['lr']}
                if epoch+1 in (config['diagnostic_epoch'],config['epochs']):
                    eval_start=time.perf_counter();scores=np.empty((len(cs),len(val_ix)),dtype=np.float64);logits=np.empty_like(scores)
                    for ci in range(len(cs)):
                        logits[ci]=logits_for(ema,cache[ci],config['eval_batch_size']);scores[ci]=sigmoid64(logits[ci])
                    metrics=summarize(labels[val_ix],scores,cs)
                    raw=sigmoid64(logits_for(model,cache[0],config['eval_batch_size']))
                    fit_scores=sigmoid64(logits_for(ema,arrays[fit_sample],config['eval_batch_size']))
                    row.update({'metrics':metrics,'raw_original_auc':auc(labels[val_ix],raw),'fit_sample_auc':auc(labels[fit_sample],fit_scores)})
                    eval_seconds+=time.perf_counter()-eval_start
                    buffer=io.BytesIO();np.savez_compressed(buffer,probabilities=scores,logits=logits,raw_original_probabilities=raw,
                        y=labels[val_ix],image_ids=np.array([rows[i]['image_id'] for i in val_ix]),group_ids=np.array([rows[i]['group_id'] for i in val_ix]),
                        condition_names=np.array([c['name'] for c in cs]),condition_weights=np.array([c['weight'] for c in cs]))
                    data=buffer.getvalue();filename=prefix+f'_e{epoch+1}_predictions.npz';(out/filename).write_bytes(data);emit_artifact(filename,data)
                    print('E1_METRICS '+json.dumps({'fold':fold,'architecture':architecture,'epoch':epoch+1,'original_auc':metrics['original_auc'],
                        'robust_auc_proxy':metrics['robust_auc_proxy'],'J04':metrics['J04'],'worst_family':metrics['worst_family'],
                        'worst_family_auc':metrics['worst_family_auc'],'fit_sample_auc':row['fit_sample_auc'],'raw_original_auc':row['raw_original_auc']}),flush=True)
                curves.append(row);(out/(prefix+'_curves.json')).write_text(json.dumps(curves,indent=2));emit_json(prefix+'_curves.json',curves)
                print('E1_EPOCH '+json.dumps({'fold':fold,'architecture':architecture,**{k:v for k,v in row.items() if k!='metrics'}}),flush=True)
            checkpoint=out/(prefix+'_final_ema.pt')
            save_bundle(checkpoint,architecture,ema,provenance,purpose='E1 development checkpoint; not final holdout evaluated',
                training_metadata={'fold':fold,'seed':seed,'epochs':config['epochs'],'protocol_sha256':environment['protocol_sha256']})
            fixture=root/'cli-fixtures'/f'fold{fold}';fixture.mkdir(parents=True,exist_ok=True)
            for i in val_ix[:16]:
                link=fixture/(rows[i]['image_id']+'.jpg')
                if not link.exists():link.symlink_to(root/rows[i]['relative_path'])
            cli_output=out/(prefix+'_cli.csv')
            command=[sys.executable,str(root/'inference_script.py'),'--test_dir',str(fixture),'--weights',str(checkpoint),'--output',str(cli_output),'--batch_size','64']
            subprocess.run(command,check=True,timeout=120);first_csv=cli_output.read_bytes();subprocess.run(command,check=True,timeout=120)
            if first_csv!=cli_output.read_bytes():raise ValueError('Same-environment CLI replay is not byte-identical')
            with cli_output.open(newline='') as f:cli_rows=list(csv.DictReader(f))
            mapped={r['image_id']:float(r['confidence']) for r in cli_rows}
            if set(mapped)!={rows[i]['image_id'] for i in val_ix[:16]}:raise ValueError('CLI ID mapping changed')
            error=max(abs(mapped[rows[i]['image_id']]-scores[0,j]) for j,i in enumerate(val_ix[:16]))
            if error>1e-5:raise ValueError('CLI/validation score mismatch: '+str(error))
            summary={'fold':fold,'architecture':architecture,'seed':seed,'epochs':config['epochs'],'fit_images':len(fit_ix),'validation_images':len(val_ix),
                'train_seconds':train_seconds,'evaluation_seconds':eval_seconds,'wall_seconds':time.perf_counter()-run_start,
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
                'final_metrics':curves[-1]['metrics'],'checkpoint_sha256':sha256(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,
                'checkpoint_remote_path':str(checkpoint),'provenance':provenance,'cli_replay_byte_identical':True,'cli_validation_max_probability_difference':error,
                'source_hashes':environment['source_hashes'],'protocol_sha256':environment['protocol_sha256'],'pool_manifest_sha256':environment['pool_manifest_sha256']}
            (out/(prefix+'_summary.json')).write_text(json.dumps(summary,indent=2));emit_json(prefix+'_summary.json',summary);completed.append(summary)
            del optimizer,scaler,model,ema;gc.collect();torch.cuda.empty_cache()
            # Release loop tensors/autograd references before the next model.
            del x,y,z,loss
        del cache;gc.collect()
    final={'runs':completed,'total_wall_seconds':time.perf_counter()-start,'holdout_used':False,'test_used':False,'T4_timing':'unmeasured',
           'interpretation':'Two-fold development screen; five-fold/second-seed confirmation remains required'}
    emit_json('complete.json',final);print('E1_COMPLETE '+json.dumps({'runs':len(completed),'wall_seconds':final['total_wall_seconds']}),flush=True)


if __name__=='__main__':main()
