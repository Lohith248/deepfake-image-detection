"""Exact binary weighted AUC with tied scores; NumPy-only remote dependency."""
import numpy as np


def auc(y, score, weights=None):
    y = np.asarray(y); score = np.asarray(score)
    if y.ndim != 1 or score.shape != y.shape or not np.isfinite(score).all() or not np.isin(y,[0,1]).all():
        raise ValueError('Invalid binary AUC input')
    w = np.ones(len(y)) if weights is None else np.asarray(weights,dtype=float)
    if w.shape != y.shape or not np.isfinite(w).all() or (w<0).any(): raise ValueError('Invalid weights')
    order = np.argsort(score,kind='stable'); y=y[order]; w=w[order]; score=score[order]
    starts = np.r_[0,np.flatnonzero(np.diff(score)) + 1]
    positive = np.add.reduceat(w*y,starts); negative=np.add.reduceat(w*(1-y),starts)
    denominator=positive.sum()*negative.sum()
    if denominator<=0:return float('nan')
    return float((positive*(np.cumsum(negative)-.5*negative)).sum()/denominator)


def sigmoid64(z):
    z=np.asarray(z,dtype=np.float64);exp=np.exp(-np.abs(z))
    return np.where(z>=0,1/(1+exp),exp/(1+exp))


def summarize(y, scores, conditions):
    scores=np.asarray(scores)
    if scores.shape!=(len(conditions),len(y)):raise ValueError('Condition score matrix mismatch')
    aucs=[auc(y,p) for p in scores];families={}
    for family in sorted({c['family'] for c in conditions} - {'original','diagnostic'}):
        ix=[i for i,c in enumerate(conditions) if c['family']==family]
        total=sum(conditions[i]['weight'] for i in ix)
        families[family]=sum(aucs[i]*conditions[i]['weight'] for i in ix)/total
    robust=sum(value*c['weight'] for value,c in zip(aucs,conditions))
    orig=aucs[0];p=scores[0];label=np.asarray(y);hard=p>=.5;tp=int(((hard==1)&(label==1)).sum())
    fp=int(((hard==1)&(label==0)).sum());fn=int(((hard==0)&(label==1)).sum());tn=int(((hard==0)&(label==0)).sum())
    clipped=np.clip(p,1e-12,1-1e-12)
    return {'original_auc':orig,'robust_auc_proxy':robust,'J04':orig+.4*robust,
            'J02':orig+.2*robust,'J10':orig+robust,'family_auc':families,
            'worst_family':min(families,key=families.get),'worst_family_auc':min(families.values()),
            'condition_auc':{c['name']:value for c,value in zip(conditions,aucs)},
            'f1_at_05':2*tp/max(1,2*tp+fp+fn),'accuracy_at_05':(tp+tn)/len(label),
            'confusion_at_05':{'tp':tp,'fp':fp,'tn':tn,'fn':fn},'predicted_fake_fraction':float(hard.mean()),
            'nll':float(-(label*np.log(clipped)+(1-label)*np.log1p(-clipped)).mean()),
            'brier':float(((p-label)**2).mean()),'saturated_original_scores':int(((p==0)|(p==1)).sum())}
