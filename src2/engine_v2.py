import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import time
import os

def train_classifier(model, train_loader, val_loader, device, epochs=50, lr=1e-3, weight=None):
    """
    Trains the IDSClassifier with Mixed Precision (AMP) for RTX 3050.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weight.to(device) if weight is not None else None)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    scaler = GradScaler() # For Mixed Precision
    
    best_val_loss = float('inf')
    best_state = None
    patience = 7
    counter = 0

    print(f"  Training Classifier on {device} (with AMP)...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Autocast for Mixed Precision
            with autocast():
                outputs = model(X)
                loss = criterion(outputs, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                with autocast():
                    outputs = model(X)
                    loss = criterion(outputs, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"    Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("    Early stopping triggered.")
                break
                
    model.load_state_dict(best_state)
    return model

def train_vae(model, train_loader, val_loader, device, epochs=30):
    """
    Trains AnomalyVAE on normal traffic only.
    """
    from models import vae_loss_function
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    best_state = None
    
    print(f"  Training VAE for Anomaly Detection (with AMP)...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, _ in train_loader:
            X = X.to(device)
            optimizer.zero_grad()
            with autocast():
                recon, mu, logvar = model(X)
                loss = vae_loss_function(recon, X, mu, logvar)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, _ in val_loader:
                X = X.to(device)
                with autocast():
                    recon, mu, logvar = model(X)
                    loss = vae_loss_function(recon, X, mu, logvar)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            
    model.load_state_dict(best_state)
    return model

def get_predictions(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            with autocast():
                outputs = torch.softmax(model(X), dim=1)
            all_probs.append(outputs.cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)

def get_vae_scores(model, loader, device):
    model.eval()
    all_scores = []
    with torch.no_grad():
        for X, _ in loader:
            X = X.to(device)
            with autocast():
                recon, _, _ = model(X)
                # Reconstruction error as anomaly score
                mse = torch.mean((recon - X)**2, dim=1)
            all_scores.append(mse.cpu().numpy())
    return np.concatenate(all_scores)
