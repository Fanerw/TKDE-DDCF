import torch


def recall(scores, labels, k):
    scores = scores
    labels = labels
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / torch.min(torch.Tensor([k]).to(hit.device), labels.sum(1).float())).mean().cpu().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}
    labels[labels>0]=1.0
    scores = scores[labels.sum(1)>0]
    labels = labels[labels.sum(1)>0]
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
       cut = cut[:, :k]
       hits = labels_float.gather(1, cut)
       #128维，每一维代表预测命中的个数
       a=hits.sum(1)
       #128维，每一维代表真实的标签推荐的总个数
       b=torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())
       c=torch.Tensor([k]).to(labels.device)
       d=labels.sum(1).float()
       # metrics['Recall@%d' % k] = \
       #     (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item()
       metrics['Recall@%d'%k]=\
           ((hits.sum(1))/labels.sum(1).float()).mean().cpu().item()
       metrics['Precision@%d'%k]=(hits.sum(1)/torch.Tensor([k]).to(labels.device)).mean().cpu().item()
       if (metrics['Recall@%d'%k]==0.0)&(metrics['Precision@%d'%k]==0.0):
           metrics['F-measure@%d' % k]=0.0
       else:
           metrics['F-measure@%d'%k]=2*metrics['Recall@%d'%k]*metrics['Precision@%d'%k]/(metrics['Recall@%d'%k]+metrics['Precision@%d'%k])

       position = torch.arange(2, 2+k)
       weights = 1 / torch.log2(position.float())
       dcg = (hits * weights.to(hits.device)).sum(1)
       idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
       ndcg = (dcg / idcg).mean()
       metrics['NDCG@%d' % k] = ndcg.cpu().item()

    return metrics