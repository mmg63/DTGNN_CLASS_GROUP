# train.py
import time
import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import KFold
from model import GATMultiHead_residual
from utils.evaluation_metrics import evaluate_multilabel_metrics, plot_confusion_matrices, plot_roc_curves
from utils.training_utils import parse_arguments, get_device, compute_pos_weight
from utils.data_loading import load_datasets, normalize_graphs, make_graphs_undirected



def main():
    args = parse_arguments()

    device = get_device()
    print(f"Using device: {device}")

    # Load datasets
    train_dataset, test_dataset = load_datasets(args.train_dataset_path, args.test_dataset_path)

    # Normalize features
    normalize_graphs(train_dataset)
    normalize_graphs(test_dataset)

    # Make undirected and remove background class
    make_graphs_undirected(train_dataset)
    make_graphs_undirected(test_dataset)
    
    pos_weight = compute_pos_weight(train_dataset, args.n_classes).to(device)
    thresholds = torch.tensor([0.3, 0.6, 0.6, 0.4], device=device)  # [green, blue, yellow, red]

    kfold = KFold(n_splits=args.k_fold, shuffle=True, random_state=42)
    models = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f"\n=== Fold {fold + 1}/{args.k_fold} ===")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  sampler=SubsetRandomSampler(train_idx), drop_last=True)
        val_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                sampler=SubsetRandomSampler(val_idx))

        model = GATMultiHead_residual(in_channels=7, out_channels=args.n_classes).to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=4e-3)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_train_acc = 0

        for epoch in range(args.epochs):
            model.train()
            train_correct = 0
            total_train = 0
            t_start = time.time()

            for batch in train_loader:
                optimizer.zero_grad()
                batch[0] = batch[0].to(device)
                batch[0].y = batch[0].y.view(-1, args.n_classes).to(device)
                # print(batch[0].y.shape)  # for debugging
                out = model(batch[0])
                loss = loss_fn(out, batch[0].y.float())
                loss.backward()
                optimizer.step()

                probs = torch.sigmoid(out)
                predicted = (probs > thresholds).float()
                correct = (predicted == batch[0].y).float()
                exact_match = correct.all(dim=1).sum().item()

                train_correct += exact_match
                total_train += batch[0].y.size(0)

            train_acc = 100 * train_correct / total_train
            print(f"Epoch {epoch+1} - Train Accuracy: {train_acc:.2f}% - Time: {time.time() - t_start:.2f}s    -    lr: {optimizer.param_groups[0]['lr']:.4f}")

            if epoch in [10, 20, 25]:
                optimizer.param_groups[0]['lr'] /= 10

        # Validation
        model.eval()
        val_correct = 0
        soft_correct = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                batch[0] = batch[0].to(device)
                batch[0].y = batch[0].y.view(-1, args.n_classes).to(device)

                out = model(batch[0])
                probs = torch.sigmoid(out)
                predicted = (probs > thresholds).float()
                correct = (predicted == batch[0].y).float()

                val_correct += correct.all(dim=1).sum().item()
                soft_correct += correct.mean().item() * batch[0].y.size(0)
                total_val += batch[0].y.size(0)

        exact_acc = 100 * val_correct / total_val
        soft_acc = 100 * soft_correct / total_val

        print(f"✅ Fold {fold+1} - Exact: {exact_acc:.2f}%, Soft: {soft_acc:.2f}%")
        models.append(model.cpu())
        fold_metrics.append({'fold': fold, 'exact_acc': exact_acc, 'soft_acc': soft_acc})

    # Test set evaluation
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    all_logits = []

    for model in models:
        model.to(device).eval()
        fold_logits = []
        with torch.no_grad():
            for batch in test_loader:
                batch[0] = batch[0].to(device)
                out = model(batch[0])
                fold_logits.append(out.cpu())
        all_logits.append(torch.cat(fold_logits, dim=0))

    avg_logits = torch.stack(all_logits).mean(dim=0)
    preds = (torch.sigmoid(avg_logits) > thresholds).float()
    y_true = torch.stack([data[0].y for data in test_dataset])

    correct = (preds == y_true).float()
    exact_acc = 100 * correct.all(dim=1).sum().item() / len(y_true)
    soft_acc = 100 * correct.mean().item()

    print("\n📊 Final Test Performance")
    print(f"✅ Exact Match: {exact_acc:.2f}%")
    print(f"✅ Soft Accuracy: {soft_acc:.2f}%")

    class_names = ['green', 'blue', 'yellow', 'red']
    conf_matrices = evaluate_multilabel_metrics(y_true, preds, label_names=class_names)
    plot_confusion_matrices(conf_matrices, class_names)
    plot_roc_curves(y_true, torch.sigmoid(avg_logits), class_names)


if __name__ == "__main__":
    main()
