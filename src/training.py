import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import time

def train_model(model, X_train, y_train, X_test, y_test, epochs=10, lr=0.01, verbose=True):
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    
    losses = {'train': [], 'test': []}
    start = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t)
            test_loss = loss_fn(test_pred, y_test_t)
        
        losses['train'].append(loss.item())
        losses['test'].append(test_loss.item())
        
        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Test Loss={test_loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t).cpu().numpy()
        test_pred = model(X_test_t).cpu().numpy()
    
    return {
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred))),
        'r2_score': float(r2_score(y_test, test_pred)),
        'training_time': time.time() - start,
        'epochs_completed': epochs,
        'best_test_loss': float(min(losses['test'])),
        'losses': losses,
        'predictions': {'test': test_pred.tolist()}
    }