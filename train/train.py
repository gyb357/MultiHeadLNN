import torch.nn as nn
import torch
import torch.optim as optim
import time
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import device
from typing import List, Tuple
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from train.utils import save_results
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score
)


def fit(
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        device: device,
        lr: float,
        epochs: int,
        patience: int,
        threshold: float,
        run: int,
        window: int,
        csv_path: str
) -> float:
    # Compute class weights for imbalanced datasets
    labels: List[int] = []
    for batch in train_loader:
        *_, y = batch
        labels.append(y)

    labels = torch.cat(labels, dim=0)
    counts = torch.bincount(labels)
    inv_freq = 1.0 / (counts + 1e-8)
    class_weights = (inv_freq / inv_freq.sum()).to(device)

    # Train modules
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience // 2)

    # Variables
    task_auc: float = 0.0
    best_auc: float = 0.0
    best_metrics: Tuple[float] = ()
    wait: int = 0
    times: List[float] = []

    # Progress bar setup
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} batches"),
        TimeElapsedColumn(),
        refresh_per_second=10,
    ) as progress:
        task = progress.add_task(f"Epoch 1/{epochs} • AUC: 0.0000", total=len(train_loader))

        for epoch in range(1, epochs + 1):
            # Reset progress bar for each epoch
            progress.reset(
                task,
                total=len(train_loader),
                completed=0,
                description=f"Epoch {epoch}/{epochs} • AUC: {task_auc:.4f}"
            )

            model.train()
            total_loss: float = 0.0
            start_time: float = time.time()

            for batch in train_loader:
                *x, y = batch
                x, y = [xi.to(device) for xi in x], y.to(device)

                optimizer.zero_grad()
                # Forward pass
                outputs = model(x)
                # Compute loss
                loss = criterion(outputs, y)

                # Backward pass
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Update total loss
                total_loss += loss.item()
                # Advance the progress bar
                progress.advance(task)

            # Average loss over the entire dataset
            train_loss = total_loss / len(train_loader.dataset)

            # Record training time
            end_time: float = time.time()
            times.append(end_time - start_time)

            # Validation
            valid_metrics = eval(model, valid_loader, device, threshold)
            valid_auc = valid_metrics[1]

            # Log the results
            task_auc = valid_auc
            progress.update(
                task,
                description=f"Epoch {epoch}/{epochs} • AUC: {task_auc:.4f}"
            )

            # Step the scheduler
            scheduler.step(valid_auc)

            # Early stopping check
            if valid_auc > best_auc:
                best_auc = valid_auc
                best_metrics = valid_metrics
                wait = 0

                # Save the best model
                if not os.path.exists('result'):
                    os.makedirs('result', exist_ok=True)

                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': epoch,
                            }, 'result/best_model.pth')
            else:
                wait += 1
                if wait >= patience:
                    print(f'Early stopping at epoch {epoch}')

                    # Log the results
                    save_results(
                        run=run,
                        window=window,
                        metrics=best_metrics,
                        train_time=sum(times) / len(times),
                        csv_path=csv_path,
                    )
                    break
    return sum(times) / len(times)


def eval(
        model: nn.Module,
        data_loader: DataLoader,
        device: device,
        threshold: float
) -> Tuple[float]:
    model.eval()
    probs: List = []
    labels: List = []

    with torch.no_grad():
        for batch in data_loader:
            *x, y = batch
            x, y = [xi.to(device) for xi in x], y.to(device)

            # Forward pass
            outputs = model(x)

            # Compute probabilities
            prob = F.softmax(outputs, dim=1)[:, 1]
            probs.extend(prob.cpu().numpy())
            labels.extend(y.cpu().numpy())

    # Convert to numpy arrays
    preds = [1 if p > threshold else 0 for p in probs]

    # Compute metrics
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    bac = balanced_accuracy_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    rec_bankruptcy = tp / (tp + fn) if (tp + fn) else 0.0
    pr_bankruptcy  = tp / (tp + fp) if (tp + fp) else 0.0
    rec_healthy    = tn / (tn + fp) if (tn + fp) else 0.0
    pr_healthy     = tn / (tn + fn) if (tn + fn) else 0.0

    micro_f1 = f1_score(labels, preds, average='micro')
    macro_f1 = f1_score(labels, preds, average='macro')

    type_1_error  = fp / (fp + tn) if (fp + tn) else 0.0
    type_2_error = fn / (tp + fn) if (tp + fn) else 0.0
    return (
        acc, auc, bac,
        tn, fp, fn, tp,
        micro_f1, macro_f1,
        type_1_error, type_2_error,
        rec_bankruptcy, pr_bankruptcy, rec_healthy, pr_healthy
    )

