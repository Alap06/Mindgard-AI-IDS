import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from .models import IDSClassifier, AnomalyVAE, vae_loss_function
from .dataset import KDDDataset
import time

def train_classifier(model, train_loader, val_loader, device, epochs=50, lr=1e-3, class_weights=None):
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 7
    
    print(f"Starting Classifier training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f}, Acc {train_acc:.2f}% | Val Loss {avg_val_loss:.4f}, Acc {val_acc:.2f}%")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_classifier.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print("Early stopping triggered.")
                break
                
    model.load_state_dict(torch.load("best_classifier.pth"))
    return model

def train_vae(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    """
    Trains VAE only on 'normal' data to learn reconstruction of legitimate traffic.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    print(f"Starting VAE training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [VAE Train]"):
            X = X.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(X)
            loss = vae_loss_function(recon_batch, X, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, _ in val_loader:
                X = X.to(device)
                recon_batch, mu, logvar = model(X)
                loss = vae_loss_function(recon_batch, X, mu, logvar)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"VAE Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f} | Val Loss {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_vae.pth")
            
    model.load_state_dict(torch.load("best_vae.pth"))
    return model

def compute_anomaly_scores(model, loader, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for X, _ in loader:
            X = X.to(device)
            recon_batch, _, _ = model(X)
            # MSE per sample
            mse = torch.mean((recon_batch - X)**2, dim=1)
            scores.extend(mse.cpu().numpy())
    return np.array(scores)
