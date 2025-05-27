
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

def evaluate_model(model, test_dataset, device, batch_size=100):
    """Evaluate model and return metrics"""
    model.eval()
    model.to(device)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = (all_predictions == all_labels).mean() * 100
    
    # Top-5 accuracy
    top5_correct = 0
    for i in range(len(all_labels)):
        top5_preds = np.argsort(all_probs[i])[-5:][::-1]
        if all_labels[i] in top5_preds:
            top5_correct += 1
    top5_accuracy = (top5_correct / len(all_labels)) * 100
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Classification report
    class_report = classification_report(all_labels, all_predictions, output_dict=True)
    
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
    print(f"Macro Avg F1: {class_report['macro avg']['f1-score']:.3f}")
    print(f"Weighted Avg F1: {class_report['weighted avg']['f1-score']:.3f}")
    
    return {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'per_class_f1': f1
    }

def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    plt.figure(figsize=(12, 10))
    
    # Show only worst performing classes
    f1_scores = cm.diagonal() / cm.sum(axis=1)
    worst_indices = np.argsort(f1_scores)[:20]
    
    cm_subset = cm[worst_indices][:, worst_indices]
    labels_subset = [class_names[i] for i in worst_indices]
    
    sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_subset, yticklabels=labels_subset)
    
    plt.title('Confusion Matrix - 20 Worst Classes')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def benchmark_model(model, device, input_size=(1, 3, 32, 32), num_runs=100):
    """Benchmark inference speed"""
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        end = time.time()
        times.append((end - start) * 1000)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\nInference Time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"FPS: {1000/avg_time:.1f}")
    
    return avg_time, std_time

def analyze_predictions(results, class_names):
    """Analyze model predictions"""
    predictions = results['predictions']
    labels = results['labels']
    cm = results['confusion_matrix']
    
    # Find most confused pairs
    cm_normalized = cm - np.diag(np.diag(cm))
    confused_pairs = []
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((i, j, cm[i, j]))
    
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("\nMost Confused Class Pairs:")
    for i, (true_idx, pred_idx, count) in enumerate(confused_pairs[:10]):
        print(f"{i+1}. {class_names[true_idx]} → {class_names[pred_idx]}: {count} times")
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    
    # Best and worst classes
    best_indices = np.argsort(per_class_acc)[-5:][::-1]
    worst_indices = np.argsort(per_class_acc)[:5]
    
    print("\nBest Performing Classes:")
    for idx in best_indices:
        print(f"- {class_names[idx]}: {per_class_acc[idx]:.2f}%")
    
    print("\nWorst Performing Classes:")
    for idx in worst_indices:
        print(f"- {class_names[idx]}: {per_class_acc[idx]:.2f}%")

# Main test function
def test_model(model_path, data_path):
    """Main function to test the model"""
    from test import CIFAR100HyperViT, load_model_safetensors, load_cifar100_dataset
    
    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = CIFAR100HyperViT(
        img_size=32, patch_size=4, embed_dim=192, depth=12,
        num_heads=3, mlp_ratio=4, drop_rate=0.1, num_classes=100
    )
    model = load_model_safetensors(model, model_path)
    
    # Load dataset
    _, test_dataset = load_cifar100_dataset(data_path)
    
    # Evaluate
    results = evaluate_model(model, test_dataset, device)
    
    # Plot confusion matrix
    plot_confusion_matrix(results['confusion_matrix'], test_dataset.fine_label_names)
    
    # Benchmark
    benchmark_model(model, device)
    
    # Analyze predictions
    analyze_predictions(results, test_dataset.fine_label_names)
    
    return results

if __name__ == "__main__":
    MODEL_PATH = 'safetensors/best_cifar100_hypervit.safetensors'
    DATA_PATH = '/Volumes/DATA/watermark_proj/data/archive/cifar-100-python'
    
    results = test_model(MODEL_PATH, DATA_PATH)
    print("\n" + "="*50)
    print("Evaluation Complete!")
    print("="*50)