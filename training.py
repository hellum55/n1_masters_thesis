from tqdm.notebook import tqdm
from typing import Tuple

from typing import Dict, List
from tqdm.notebook import tqdm

from typing import Tuple
import torch

from torch.utils.tensorboard import SummaryWriter

def execute_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Executes a single training epoch and returns average loss, top-1 and top-5 accuracy.
    """

    model.train()
    train_loss, top1_acc, top5_acc = 0.0, 0.0, 0.0

    for X, y, _ in tqdm(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Top-1 accuracy
        top1 = torch.argmax(y_pred, dim=1)
        top1_acc += (top1 == y).sum().item() / y.size(0)

        # Top-5 accuracy
        top5_preds = torch.topk(y_pred, k=5, dim=1).indices
        match_top5 = top5_preds.eq(y.view(-1, 1))  # shape: [batch, 5]
        top5_acc += match_top5.any(dim=1).float().sum().item() / y.size(0)

    # Average over all batches
    num_batches = len(dataloader)
    train_loss /= num_batches
    top1_acc /= num_batches
    top5_acc /= num_batches

    return train_loss, top1_acc, top5_acc

def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Evaluates model performance on a validation/test set.

    Returns:
        avg_loss, top1_accuracy, top5_accuracy
    """

    model.eval()
    eval_loss, top1_acc, top5_acc = 0.0, 0.0, 0.0

    with torch.inference_mode():
        for X, y, _ in dataloader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            eval_loss += loss.item()

            # Top-1 accuracy
            top1 = torch.argmax(y_pred, dim=1)
            top1_acc += (top1 == y).sum().item() / y.size(0)

            # Top-5 accuracy
            top5_preds = torch.topk(y_pred, k=5, dim=1).indices
            match_top5 = top5_preds.eq(y.view(-1, 1))  # shape: [batch, 5]
            top5_acc += match_top5.any(dim=1).float().sum().item() / y.size(0)

    # Averages
    num_batches = len(dataloader)
    eval_loss /= num_batches
    top1_acc /= num_batches
    top5_acc /= num_batches

    return eval_loss, top1_acc, top5_acc

def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    eval_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    writer: SummaryWriter  # ✅ New: tensorboard writer
) -> Dict[str, List]:
    """
    Trains and evaluates a model over multiple epochs and logs top-1 and top-5 accuracy.
    """

    session = {
        'loss': [],
        'accuracy': [],
        'top5_accuracy': [],
        'eval_loss': [],
        'eval_accuracy': [],
        'eval_top5_accuracy': []
    }

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')

        # Training
        train_loss, train_top1, train_top5 = execute_epoch(
            model, train_dataloader, optimizer, loss_fn, device
        )

        # Evaluation
        eval_loss, eval_top1, eval_top5 = evaluate(
            model, eval_dataloader, loss_fn, device
        )

        # Log metrics to TensorBoard ✅
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", eval_loss, epoch)
        writer.add_scalar("Accuracy/Train_top1", train_top1, epoch)
        writer.add_scalar("Accuracy/Val_top1", eval_top1, epoch)
        writer.add_scalar("Accuracy/Train_top5", train_top5, epoch)
        writer.add_scalar("Accuracy/Val_top5", eval_top5, epoch)

        # Print logs
        print(
            f'loss: {train_loss:.4f} - top1: {train_top1:.4f} - top5: {train_top5:.4f} '
            f'- eval_loss: {eval_loss:.4f} - eval_top1: {eval_top1:.4f} - eval_top5: {eval_top5:.4f}'
        )

        # Save to session log
        session['loss'].append(train_loss)
        session['accuracy'].append(train_top1)
        session['top5_accuracy'].append(train_top5)
        session['eval_loss'].append(eval_loss)
        session['eval_accuracy'].append(eval_top1)
        session['eval_top5_accuracy'].append(eval_top5)

    return session