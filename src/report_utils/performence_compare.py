import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics.classification import MulticlassAccuracy

# optional: tqdm for a nice progress bar (falls back to simple prints if missing)
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


def get_validation_loader(
    path,
    batch_size,
    mean=(0.5025, 0.4846, 0.5003),
    std=(0.1574, 0.1490, 0.1549),
    num_workers=4,
    pin_memory=True,
):
    transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=tuple(float(m) for m in mean),
                             std=tuple(float(s) for s in std)),
    ])
    val_ds = datasets.ImageFolder(path, transform)
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return val_dl


@torch.inference_mode()
def model_performance(
    model,
    path='/home/temp/MyDir/Projects/aspdfpwjfpwejfwpefwef/Drons/HighRPD_parsed/val',
    batch_size=16,
    mean=(0.5025, 0.4846, 0.5003),
    std=(0.1574, 0.1490, 0.1549),
    device='cpu',
    top_k=1,
    verbose=True,
):
    """
    Evaluate a classifier on an ImageFolder validation set.

    Prints a concise report and returns:
        {
            'accuracy': float,
            'num_classes': int,
            'num_samples': int,
            'avg_latency_ms': float,
            'throughput_img_s': float
        }
    """
    device = torch.device(device)
    model = model.to(device).eval()

    val_dl = get_validation_loader(
        path=path, batch_size=batch_size, mean=mean, std=std,
        num_workers=4, pin_memory=(device.type == 'cuda')
    )
    num_classes = len(val_dl.dataset.classes)

    metric = MulticlassAccuracy(num_classes=num_classes, top_k=top_k).to(device)

    # Pretty header
    if verbose:
        print("=" * 60)
        print(f"Validation on: {path}")
        print(f"Device: {device} | Batch size: {batch_size} | Classes: {num_classes}")
        print(f"Top-{top_k} accuracy")
        print("-" * 60)

    total_time = 0.0
    total_seen = 0

    iterator = val_dl
    if verbose and _HAS_TQDM:
        iterator = tqdm(val_dl, desc="Evaluating", unit="batch")

    for X, y in iterator:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        start = time.perf_counter()
        logits = model(X)
        # For top-1, argmax is fine; MulticlassAccuracy also accepts logits directly.
        preds = logits.argmax(dim=1) if top_k == 1 else logits
        metric.update(preds, y)
        batch_time = time.perf_counter() - start

        total_time += batch_time
        total_seen += X.size(0)

        if verbose and not _HAS_TQDM:
            # simple, non-tqdm progress every ~10 batches
            if total_seen // max(1, batch_size) % 10 == 0:
                running_acc = metric.compute().item()
                print(f"  Seen {total_seen:5d} imgs | Acc: {running_acc:6.3%}")

    acc = metric.compute().item()
    avg_latency_ms = (total_time / max(1, len(val_dl))) * 1000.0
    throughput = total_seen / max(1e-9, total_time)

    if verbose:
        print("-" * 60)
        print(f"Samples        : {total_seen}")
        print(f"Accuracy       : {acc:6.3%}")
        print(f"Avg batch time : {avg_latency_ms:7.2f} ms")
        print(f"Throughput     : {throughput:7.2f} img/s")
        print("=" * 60)

    return {
        "accuracy": acc,
        "num_classes": num_classes,
        "num_samples": total_seen,
        "avg_latency_ms": avg_latency_ms,
        "throughput_img_s": throughput,
    }


