import torch

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    eval_losses = {'total_loss': 0.0, 'chamfer_loss': 0.0,
                   'transform_loss': 0.0, 'transform_chamfer_loss': 0.0}

    with torch.no_grad():
        for batch_idx, (source, target, gt_transform) in enumerate(dataloader):
            source, target, gt_transform = source.to(device), target.to(device), gt_transform.to(device)
            pred_transform, selected_source, selected_target, _ = model(source, target)
            losses = loss_fn(pred_transform, gt_transform, source, target)
            for k, v in losses.items():
                eval_losses[k] += v.item()
    for k in eval_losses.keys():
        eval_losses[k] /= len(dataloader)

    return eval_losses