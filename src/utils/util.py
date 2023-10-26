# compute classify acc
def compute_acc(output, target, topk=(1, 5)):
    maxk = max(topk)
    sample_num = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred)).contiguous()
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True).item()
        res.append((correct_k / sample_num if sample_num > 0 else 0))
    return res


# compute mertics
