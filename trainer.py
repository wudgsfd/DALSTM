import torch
import torch.optim as optim
import pandas as pd
import os
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_and_evaluate(model, model_name, criterion, num_epochs, batch_size, train_X, train_y, val_X, val_y,
                       test_X, test_y, save_dir, timestamp, device, concept_ids=None):
    start_time = datetime.now()
    print(f"\n{model_name} 开始训练时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    history = {
        'loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'epoch_time': []
    }

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in range(num_epochs):
        epoch_start = datetime.now()

        model.train()
        running_loss = 0.0
        for i in range(0, len(train_X), batch_size):
            inputs = train_X[i:i + batch_size].to(device)
            targets = train_y[i:i + batch_size].to(device)

            # 处理特殊模型的输入需求
            if model_name == 'Graph-based KT' and concept_ids is not None:
                batch_concepts = concept_ids[i:i + batch_size].to(device)
                optimizer.zero_grad()
                outputs = model(inputs, batch_concepts)
            else:
                optimizer.zero_grad()
                outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        with torch.no_grad():
            # 验证集评估
            if model_name == 'Graph-based KT' and concept_ids is not None:
                val_outputs = model(val_X.to(device), concept_ids[:len(val_X)].to(device))
            else:
                val_outputs = model(val_X.to(device))

            val_loss = criterion(val_outputs, val_y.to(device))
            val_probs = val_outputs.softmax(dim=1).cpu().numpy()
            val_preds = val_outputs.argmax(dim=1).cpu().numpy()
            val_targets = val_y.cpu().numpy()

            # 计算验证集指标
            pre = precision_score(val_targets, val_preds, average='macro')
            rec = recall_score(val_targets, val_preds, average='macro')
            f1 = f1_score(val_targets, val_preds, average='macro')
            roc_auc = roc_auc_score(val_targets, val_probs, multi_class='ovr', average='macro')
            val_acc = accuracy_score(val_targets, val_preds)

            # 记录指标
            avg_train_loss = running_loss / (len(train_X) // batch_size)
            history['loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss.item())
            history['precision'].append(pre)
            history['recall'].append(rec)
            history['f1'].append(f1)
            history['roc_auc'].append(roc_auc)
            history['val_accuracy'].append(val_acc)

        # 计算训练集准确率
        with torch.no_grad():
            if model_name == 'Graph-based KT' and concept_ids is not None:
                train_outputs = model(train_X.to(device), concept_ids[:len(train_X)].to(device))
            else:
                train_outputs = model(train_X.to(device))

            train_preds = train_outputs.argmax(dim=1).cpu().numpy()
            train_targets = train_y.cpu().numpy()
            train_acc = accuracy_score(train_targets, train_preds)
            history['train_accuracy'].append(train_acc)

        # 计算本轮epoch耗时
        epoch_end = datetime.now()
        epoch_duration = (epoch_end - epoch_start).total_seconds()
        history['epoch_time'].append(epoch_duration)

        # 更新学习率
        scheduler.step(val_loss)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], 耗时: {epoch_duration:.2f}秒, '
            f'Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}, '
            f'Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}, '
            f'Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}'
        )

    # 记录模型训练结束时间并计算总耗时
    end_time = datetime.now()
    total_duration = end_time - start_time
    print(f"{model_name} 训练结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{model_name} 总训练时间: {str(total_duration).split('.')[0]} (时:分:秒)")

    # 测试集评估
    with torch.no_grad():
        if model_name == 'Graph-based KT' and concept_ids is not None:
            test_outputs = model(test_X.to(device), concept_ids[len(train_X) + len(val_X):].to(device))
        else:
            test_outputs = model(test_X.to(device))

        test_loss = criterion(test_outputs, test_y.to(device))
        test_probs = test_outputs.softmax(dim=1).cpu().numpy()
        test_preds = test_outputs.argmax(dim=1).cpu().numpy()
        test_targets = test_y.cpu().numpy()

        test_pre = precision_score(test_targets, test_preds, average='macro')
        test_rec = recall_score(test_targets, test_preds, average='macro')
        test_f1 = f1_score(test_targets, test_preds, average='macro')
        test_roc_auc = roc_auc_score(test_targets, test_probs, multi_class='ovr', average='macro')
        test_acc = accuracy_score(test_targets, test_preds)

        print(
            f'Test Loss: {test_loss.item():.4f}, Precision: {test_pre:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}, '
            f'ROC AUC: {test_roc_auc:.4f}, Test Accuracy: {test_acc:.4f}'
        )

    # 保存指标
    metrics_df = pd.DataFrame(history)
    metrics_csv_path = os.path.join(save_dir, f'{model_name}_training_metrics_{timestamp}.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)

    # 保存预测结果
    predictions = {
        'predicted_probabilities': [prob.tolist() for prob in test_probs],
        'actuals': test_targets.tolist()
    }
    predictions_df = pd.DataFrame(predictions)
    predictions_csv_path = os.path.join(save_dir, f'{model_name}_predictions_{timestamp}.csv')
    predictions_df.to_csv(predictions_csv_path, index=False)

    return history, predictions, total_duration