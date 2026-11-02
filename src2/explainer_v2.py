import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def run_shap_analysis(model, X_train, X_test, feature_names, device, output_dir: Path):
    """
    Standard SHAP analysis for Deep Learning models.
    Uses GradientExplainer (standard for PyTorch).
    """
    print("\n  [Phase 7] Running SHAP Explainability (Standard)...")
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # We use a small subset of test for speed, as DeepExplainer is heavy
    n_samples = min(200, len(X_test))
    test_subset = torch.tensor(X_test[:n_samples], dtype=torch.float32).to(device)
    background = torch.tensor(X_train[:100], dtype=torch.float32).to(device)
    
    try:
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(test_subset)
        
        # Plotting
        plt.figure(figsize=(12, 8))
        # shap_values is a list for multi-class. We plot for the first class (e.g., DoS) or summary.
        # Summary plot over all classes
        shap.summary_plot(shap_values, test_subset.cpu().numpy(), feature_names=feature_names, show=False)
        plot_path = output_dir / "shap_summary.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ SHAP summary plot saved: {plot_path}")
        return str(plot_path)
        
    except Exception as e:
        print(f"    ⚠ SHAP error: {e}")
        print("    Falling back to Weight-based Importance...")
        return None

def generate_performance_plots(all_true, all_preds, le, output_dir: Path):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(all_true, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    
    plot_path = output_dir / "confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close()
    return str(plot_path)
