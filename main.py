import torch
from data_preprocess import create_graph
from model import SampledGraphormer
from train_eval import Trainer
from utils import save_results, plot_roc_curve, plot_pr_curve
from config import Config

if __name__ == "__main__":
    torch.cuda.empty_cache()
    train_loader, val_loader, test_loader, train_data, val_data, test_data = create_graph()

    model = SampledGraphormer(
        input_dim=train_data.x.size(1),
        cell_dim=train_data.edge_attr.size(1)
    ).to(Config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    trainer = Trainer(model, train_loader, val_loader, test_loader, optimizer, criterion)

    for epoch in range(Config.epochs):
        train_metrics = trainer.train_epoch()
        val_metrics = trainer.evaluate('val')
        scheduler.step(val_metrics['loss'])
        save_results(epoch, train_metrics, val_metrics, best_val_loss=trainer.best_val_loss)

        print(f"Epoch {epoch + 1}/{Config.epochs}")
        print(
            f"Train - Loss: {train_metrics['loss']:.4f} AUC: {train_metrics['auc']:.4f} AP: {train_metrics['ap']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f} AUC: {val_metrics['auc']:.4f} AP: {val_metrics['ap']:.4f}")
        print(
            f"Threshold: {val_metrics['threshold']:.4f} F1: {val_metrics['f1']:.4f} Precision: {val_metrics['precision']:.4f} Recall: {val_metrics['recall']:.4f}\n")

    trainer.load_best_model()
    test_metrics = trainer.evaluate('test')
    print(f"Test Results - AUC: {test_metrics['auc']:.4f} AP: {test_metrics['ap']:.4f}")
    print(
        f"Threshold: {test_metrics['threshold']:.4f} F1: {test_metrics['f1']:.4f} Precision: {test_metrics['precision']:.4f} Recall: {test_metrics['recall']:.4f}")

    trainer.save_test_results("detailed_test_results.csv")

    plot_roc_curve(trainer.test_labels, trainer.test_preds)
    plot_pr_curve(trainer.test_labels, trainer.test_preds)
