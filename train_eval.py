import os
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve, average_precision_score
from config import Config


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, criterion):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optim = optimizer
        self.criterion = criterion
        self.test_preds = []
        self.test_labels = []
        self.test_edge_info = []
        self.best_val_loss = float('inf')
        self.best_val_auc = 0.0
        self.best_threshold = 0.5
        self.best_model_path = "best_model.pth"

    def _find_optimal_threshold(self, pred, target):
        probs = torch.sigmoid(pred).numpy()
        fpr, tpr, thresholds = roc_curve(target.numpy(), probs)
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        return thresholds[best_idx]

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        for batch in self.train_loader:
            self.optim.zero_grad()
            batch = batch.to(Config.device)
            pred, labels = self.model(batch, mode='train')
            labels = labels.float().to(Config.device)
            loss = self.criterion(pred, labels)
            loss.backward()
            self.optim.step()
            total_loss += loss.item()
            all_preds.append(pred.detach().cpu())
            all_labels.append(labels.cpu())
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        return self._compute_metrics(total_loss, all_preds, all_labels, len(self.train_loader))

    def evaluate(self, mode):
        self.model.eval()
        loader = self.val_loader if mode == 'val' else self.test_loader
        total_loss = 0
        all_preds = []
        all_labels = []
        all_edge_info = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(Config.device)
                pred, labels = self.model(batch, mode=mode)
                loss = self.criterion(pred, labels.float())
                total_loss += loss.item()
                all_preds.append(pred.cpu())
                all_labels.append(labels.cpu())
                if hasattr(batch, 'edge_info'):
                    all_edge_info.extend(batch.edge_info)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        metrics = self._compute_metrics(total_loss, all_preds, all_labels, len(loader))

        if mode == 'test':
            self.test_preds = torch.sigmoid(all_preds).numpy()
            self.test_labels = all_labels.numpy().astype(int)
            self.test_edge_info = all_edge_info

            # 验证标签值
            unique_labels = np.unique(self.test_labels)
            if set(unique_labels) != {0, 1}:
                self.test_labels = (self.test_labels > 0.5).astype(int)

        if mode == 'val':
            val_loss = metrics['loss']
            val_auc = metrics['auc']
            if val_loss < self.best_val_loss:
                self.best_val_auc = val_auc
                self.best_val_loss = val_loss
                self.best_threshold = metrics['threshold']
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"保存最佳模型，AUC: {val_auc:.4f}, 损失: {val_loss:.4f}, 阈值: {self.best_threshold:.4f}")

        return metrics

    def _compute_metrics(self, total_loss, pred, target, num_batches):
        probs = torch.sigmoid(pred).numpy()
        metrics = {
            'loss': total_loss / num_batches,
            'auc': roc_auc_score(target, probs),
            'ap': average_precision_score(target, probs),
        }

        if num_batches != len(self.val_loader):
            metrics.update({
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'threshold': 0.5
            })
        else:  # 验证阶段
            self.best_threshold = self._find_optimal_threshold(pred, target)
            metrics['threshold'] = self.best_threshold

            pred_label = (probs > self.best_threshold).astype(int)
            metrics.update({
                'f1': f1_score(target, pred_label),
                'precision': precision_score(target, pred_label),
                'recall': recall_score(target, pred_label)
            })

        return metrics

    def save_test_results(self, filename="detailed_test_results.csv"):
        # 确保数据长度一致
        min_length = min(len(self.test_labels), len(self.test_preds), len(self.test_edge_info))
        if min_length == 0:
            print("警告：测试结果为空，无法保存")
            return

        self.test_labels = self.test_labels[:min_length]
        self.test_preds = self.test_preds[:min_length]
        self.test_edge_info = self.test_edge_info[:min_length]

        # 解包药物对和细胞系信息
        drug1_list = [info[0] for info in self.test_edge_info]
        drug2_list = [info[1] for info in self.test_edge_info]
        cell_line_list = [info[2] for info in self.test_edge_info]

        df = pd.DataFrame({
            'true_label': self.test_labels,
            'pred_prob': self.test_preds,
            'pred_label': (self.test_preds > self.best_threshold).astype(int),
            'drug1': drug1_list,
            'drug2': drug2_list,
            'cell_line': cell_line_list
        })
        df.to_csv(filename, index=False)
        print(f"测试结果已保存到 {filename}")

    def load_best_model(self):
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path))
            print("加载最佳模型")
        else:
            print("警告：未找到保存的最佳模型")