import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
import time
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    """
    Focal Loss to handle class imbalance.
    Downweights easy (majority class) examples so the model focuses
    on hard (minority class) examples.
    gamma=2.0 is standard; alpha handled via class_weights.
    """
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_model(model, X_train, X_full_train, y_train, X_test, X_full_test, y_test,
                epochs=30, lr=0.001, verbose=True, batch_size=64, class_weights=None):

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_full_train_t = torch.tensor(X_full_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    X_full_test_t = torch.tensor(X_full_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_t, X_full_train_t, y_train_t)

    # Implement perfectly balanced Weighted Batch Sampling
    class_counts = np.bincount(y_train)
    # Weights for each sample
    sample_weights = 1.0 / class_counts[y_train]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    # When using sampler, shuffle must be False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

    # Separate parameter groups: quantum params get a much lower lr
    # to preserve quantum phase stability; classical gets normal lr with L2 decay
    quantum_params = []
    classical_params = []
    for name, param in model.named_parameters():
        if 'qlayer' in name:
            quantum_params.append(param)
        else:
            classical_params.append(param)

    optimizer = optim.AdamW([
        {'params': quantum_params, 'lr': 1e-4, 'weight_decay': 0.0},
        {'params': classical_params, 'lr': lr, 'weight_decay': 1e-3}
    ])

    # CosineAnnealingLR gives a smooth, data-driven lr decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    loss_fn = FocalLoss(weight=class_weights, gamma=2.0)

    # Logit Calibration Boost for Inference
    # We heavily penalize false negatives on the minority classes.
    calibration_boost = torch.tensor([2.0, 2.0, 1.0], dtype=torch.float32)

    losses = {'train': [], 'test': []}
    best_acc = 0.0
    start = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_X_full, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X, batch_X_full)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            # Gradient clipping prevents exploding gradients in quantum params
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            test_pred_logits = model(X_test_t, X_full_test_t)
            test_loss = loss_fn(test_pred_logits, y_test_t)
            
            # Apply Inference Calibration
            calibrated_logits = test_pred_logits * calibration_boost
            _, test_pred_class = torch.max(calibrated_logits, 1)
            
            acc = accuracy_score(y_test, test_pred_class.numpy())

        losses['train'].append(train_loss)
        losses['test'].append(test_loss.item())

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{model.__class__.__name__}_best.pt")

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            current_f1 = f1_score(y_test, test_pred_class.numpy(), average='macro', zero_division=0)
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Test Loss={test_loss.item():.4f}, "
                  f"Acc={acc*100:.2f}%, F1={current_f1*100:.2f}%")

    model.eval()
    with torch.no_grad():
        train_logits = model(X_train_t, X_full_train_t) * calibration_boost
        test_logits = model(X_test_t, X_full_test_t) * calibration_boost
        _, train_pred_class = torch.max(train_logits, 1)
        _, test_pred_class = torch.max(test_logits, 1)

    test_acc = accuracy_score(y_test, test_pred_class.numpy())
    report = classification_report(y_test, test_pred_class.numpy(), zero_division=0)
    print("\nFinal Classification Report:\n", report)

    return {
        'train_acc': accuracy_score(y_train, train_pred_class.numpy()),
        'test_acc': test_acc,
        'classification_report': report,
        'training_time': time.time() - start,
        'epochs_completed': epochs,
        'best_accuracy': best_acc,
        'losses': losses,
    }