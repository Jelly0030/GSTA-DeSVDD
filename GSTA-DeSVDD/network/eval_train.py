import torch


def eval_train(net, c, dataloader, device, R):
    scores = []
    labels = []
    net.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            z = net(x)
            z = torch.tensor(z.clone().detach())
            c = torch.tensor(c.clone().detach())
            score = torch.sum((z - c) ** 2, dim=-1)
            scores.append(score.detach().cpu())
            labels.append(y.cpu())

    labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()
    preds = (scores < R).astype(int)
    return labels, preds, scores

def eval_test(net, c, dataloader, device, R):
    scores = []
    labels = []
    net.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            z = net(x)
            score = torch.sum((z - c) ** 2, dim=-1)
            scores.append(score.detach().cpu())
            labels.append(y.cpu())
            loss = torch.mean(score)
            total_loss += loss.item()
    labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()
    preds = (scores < R).astype(int)
    return labels, preds, scores, total_loss



