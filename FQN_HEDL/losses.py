import torch
import torch.nn.functional as F

def get_unique(a):
    a = a.transpose(1,2)
    b=torch.zeros(a.shape)
    for i in range(len(a)):
        c = torch.unique(a[i], dim=0)
        b[i][:c.shape[0]] = c

    return b.transpose(1,2)

def get_dir_num(a):
    a = a.transpose(1,2)
    c=torch.zeros(a.shape[0])
    for i in range(len(a)):
        d = torch.unique(a[i], dim=0)
        c[i]=d.shape[0]

    return c

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    # ← 恢复KL正则项，这是EDL能产生正确不确定度的关键
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)

    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = torch.exp(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss

def edl_try(
    W,output, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + W/num_classes
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss




def get_belief(features, weight, W, num_classes, device):
    features = features.to(device)
    # weight [num_classes,fs]
    weight = relu_evidence(weight)
    # evidence [fs]
    evidence = torch.sum(weight, axis=0)
    # mask [num_classes,fs]
    mask = (weight > (1.0*evidence/num_classes).unsqueeze(0)).float()
    # item_num [fs]
    item_num = torch.sum(mask, axis=0)
    # item_num [num_classes,fs]
    item_num = item_num.unsqueeze(0).expand(num_classes, -1)

    sharp_belief_item = torch.eq(item_num, 1)
    vague_belief_item = ~sharp_belief_item

    # item = [bs, num_classes, fs]
    # 表示每个特征到每一类的证据百分比
    sharp_belief_item = sharp_belief_item*mask
    vague_belief_item = vague_belief_item*mask / torch.max(torch.ones(item_num.shape).to(device),item_num)

    # 计算证据到信念
    # 累加乘积
    # 在超意见到多项式分布的投影中，基本信念只分配给单例

    # 以下使用全连接层方式计算
    henn_sharp_fc = torch.nn.Linear(len(features[1]), num_classes)
    henn_sharp_fc.weight.data = sharp_belief_item.to(device)
    henn_sharp_fc.bias.data = torch.zeros(num_classes).to(device)
    sharp_belief = henn_sharp_fc(features)

    henn_vague_fc = torch.nn.Linear(len(features[1]), num_classes)
    henn_vague_fc.weight.data = vague_belief_item.to(device)
    henn_vague_fc.bias.data = torch.zeros(num_classes).to(device)
    vague_belief = henn_vague_fc(features)

    uncertainty = torch.full((num_classes,), W/num_classes).to(device)
    return sharp_belief, vague_belief, uncertainty

def get_henn_fc(features, weight, W, num_classes, device):
    
    # weight = relu_evidence(weight)
    # # weight = softplus_evidence(weight)
    # # weight = exp_evidence(weight)
    # evidence = torch.sum(weight, axis=0)
    # mask = (weight > (1.0*evidence/num_classes).unsqueeze(0)).float()
    # # mask = (weight > (1.0*evidence/torch.sum(weight > 0,axis=0)).unsqueeze(0)).float()

    mask = (weight > 0).float()
    item_num = torch.sum(mask, axis=0)
    item_num = item_num.unsqueeze(0).expand(num_classes, -1)
    weight = mask / torch.max(torch.ones(item_num.shape).to(device),item_num)

    henn_fc = torch.nn.Linear(len(features[1]), num_classes)
    henn_fc.weight.data = weight.to(device)
    henn_fc.bias.data = torch.full((num_classes,), W/num_classes).to(device)
    features = features.to(device)
    
    return henn_fc(features)


def hedl_loss(func, y, features, weight, W, num_classes, epoch_num, annealing_step, device=None, outputs = None):
    y = y.to(device)
    alpha = get_henn_fc(features, weight, W, num_classes, device)
    if outputs is not None:
        outputs.data = alpha.data.clone().detach()
        S = torch.sum(outputs, dim=1, keepdim=True).to(device)
        A = torch.sum(y * (func(S) - func(outputs)), dim=1, keepdim=True)
        return A
    S = torch.sum(alpha, dim=1, keepdim=True).to(device)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
    return A


def edl_HENN(
    W, features, weight, target, epoch_num, num_classes, annealing_step, device=None, outputs = None
):
    if not device:
        device = get_device()

    loss = torch.mean(
        hedl_loss(
            torch.digamma, target, features, weight, W, num_classes, epoch_num, annealing_step, device, outputs
        )
    )
    return loss

def edl_HENN_log(
    W, features, weight, target, epoch_num, num_classes, annealing_step, device=None, outputs = None
):
    if not device:
        device = get_device()

    loss = torch.mean(
        hedl_loss(
            torch.log, target, features, weight, W, num_classes, epoch_num, annealing_step, device, outputs
        )
    )
    return loss

def edl_HENN_mse(
    W, features, fc_weight, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = get_device()

    # dir_num, sharp_belief,vague_belief = get_belief(features, W, num_classes, device)

    sum_belief = get_henn_fc(features, fc_weight, W, num_classes, device)
    alpha = sum_belief.to(device)
    loglikelihood = loglikelihood_loss(target, alpha, device=device)
    loss = torch.mean(loglikelihood)
    return loss

