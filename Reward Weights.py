import os
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# 设置随机种子以确保结果可复现
def set_seed(seed=105):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
set_seed()

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# -------------------------- 数据集读取与预处理 --------------------------
def load_and_preprocess_data(data_path="skill_builder_data_corrected_collapsed.csv"):
    # 读取数据
    df = pd.read_csv(data_path, encoding='latin1')
    print("原始数据前5行:")
    print(df.head())
    
    # 处理标签列（correct）
    valid_values = [0, 1]
    df['correct'] = df['correct'].apply(lambda x: 0 if x not in valid_values else x)
    print(f"标签列（correct）唯一值: {df['correct'].unique()}")
    
    # 定义特征列表及名称映射（用于可视化）
    features = ['user_id', 'problem_id', 'skill_id', 'overlap_time', 'attempt_count']
    feature_names = {
        'user_id': '用户ID',
        'problem_id': '题目ID',
        'skill_id': '技能ID',
        'overlap_time': '重叠时间',
        'attempt_count': '尝试次数'
    }
    print(f"使用的特征: {features}")
    
    # 检查特征列是否存在
    missing_columns = [col for col in features if col not in df.columns]
    if missing_columns:
        raise ValueError(f"数据中缺少必要特征列: {missing_columns}")
    
    # 移除特征列中的缺失值
    df = df.dropna(subset=features)
    print(f"预处理后的数据量: {len(df)} 条")
    
    # 类别特征编码
    label_encoders = {}
    for col in ['user_id', 'problem_id', 'skill_id']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"编码特征 {col}，类别数: {len(le.classes_)}")
    
    # 特征归一化
    scalers = {}
    for feature in features:
        scaler = MinMaxScaler()
        df[feature] = scaler.fit_transform(df[feature].values.reshape(-1, 1)).flatten()
        scalers[feature] = scaler
    
    # 按用户、技能、时间排序
    df = df.sort_values(by=['user_id', 'skill_id', 'overlap_time', 'attempt_count'])
    
    # 创建序列数据（滑动窗口法）
    def split_data(data, time_step):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data.iloc[i:i + time_step][features].values)
            y_val = data.iloc[i + time_step]['correct']
            y.append(y_val)
        return np.array(X), np.array(y)
    
    # 序列长度
    time_step = 50  
    dataX, datay = split_data(df, time_step=time_step)
    print(f"序列数据形状: X={dataX.shape}, y={datay.shape}")
    
    # 【数据集比例控制】调整此处控制使用的数据量
    data_percentage = 0.2  # 例如：0.5表示使用50%的数据，1.0表示全量数据
    num_samples = int(dataX.shape[0] * data_percentage)
    dataX = dataX[:num_samples]
    datay = datay[:num_samples]
    print(f"使用 {data_percentage*100}% 的数据，最终形状: X={dataX.shape}, y={datay.shape}")
    
    # 数据集划分
    def train_val_test_split(dataX, datay, shuffle=True):
        if shuffle:
            indices = np.arange(len(dataX))
            np.random.shuffle(indices)
            dataX, datay = dataX[indices], datay[indices]
        total = len(dataX)
        train_idx = int(total * 0.6)
        val_idx = int(total * 0.8)
        return (dataX[:train_idx], datay[:train_idx],
                dataX[train_idx:val_idx], datay[train_idx:val_idx],
                dataX[val_idx:], datay[val_idx:])
    
    train_X, train_y, val_X, val_y, test_X, test_y = train_val_test_split(dataX, datay, shuffle=True)
    print(f"训练集: {len(train_X)} 验证集: {len(val_X)} 测试集: {len(test_X)}")
    
    # 转换为PyTorch张量
    train_X = torch.tensor(train_X, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1).to(device)
    val_X = torch.tensor(val_X, dtype=torch.float32).to(device)
    val_y = torch.tensor(val_y, dtype=torch.float32).unsqueeze(1).to(device)
    test_X = torch.tensor(test_X, dtype=torch.float32).to(device)
    test_y = torch.tensor(test_y, dtype=torch.float32).unsqueeze(1).to(device)
    
    return {
        'train_X': train_X, 'train_y': train_y,
        'val_X': val_X, 'val_y': val_y,
        'test_X': test_X, 'test_y': test_y,
        'features': features,
        'feature_names': feature_names,
        'label_encoders': label_encoders,
        'scalers': scalers,
        'time_step': time_step
    }


# -------------------------- 模型定义 --------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # 生成位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 修复：检查是否已存在'pe'缓冲区，如果存在则不重复注册
        if not hasattr(self, 'pe'):
            self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        # 如果序列长度超过预定义的max_len，则动态生成新的位置编码
        if seq_len > self.pe.size(1):
            new_pe = torch.zeros(seq_len, self.d_model, device=x.device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * 
                                (-np.log(10000.0) / self.d_model))
            new_pe[:, 0::2] = torch.sin(position * div_term)
            new_pe[:, 1::2] = torch.cos(position * div_term)
            new_pe = new_pe.unsqueeze(0)
            return x + new_pe
        return x + self.pe[:, :seq_len, :].to(x.device)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.5):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        k = self.k_linear(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        output = torch.matmul(attention, v).transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        output = self.out_linear(output)
        return output, attention


class CNN_LSTM_MultiHeadAttention(nn.Module):
    def __init__(self, conv_input, input_size, hidden_size, num_layers, output_size=1, 
                 dropout_prob=0.5, num_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=conv_input, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=conv_input, out_channels=conv_input, kernel_size=3, padding=1)
        self.res_conv = nn.Conv1d(in_channels=conv_input, out_channels=conv_input, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=conv_input, out_channels=conv_input, kernel_size=5, padding=2)
        
        self.lstm = nn.LSTM(conv_input, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.multi_head_attention = MultiHeadAttentionLayer(hidden_size * 2, num_heads, dropout_prob)
        self.positional_encoding = PositionalEncoding(hidden_size * 2)
        
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, feature_weights=None):
        if feature_weights is not None:
            feature_weights = feature_weights.to(x.device)
            x = x * feature_weights
        
        x_permuted = x.permute(0, 2, 1)
        conv_out = self.conv1(x_permuted)
        conv_out = F.relu(conv_out)
        conv_out = self.conv2(conv_out)
        conv_out = F.relu(conv_out)
        conv_out = self.res_conv(conv_out) + conv_out
        conv_out = self.conv3(conv_out)
        conv_out = F.relu(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        lstm_out, _ = self.lstm(conv_out, (h0, c0))
        
        lstm_out = self.positional_encoding(lstm_out)
        lstm_out = self.dropout(lstm_out)
        attn_out, _ = self.multi_head_attention(lstm_out, lstm_out, lstm_out)
        
        out = attn_out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


# -------------------------- 训练与评估函数 --------------------------
def train_and_evaluate(model, model_name, criterion, num_epochs, batch_size, 
                       train_X, train_y, val_X, val_y, test_X, test_y, 
                       save_dir, timestamp, feature_weights=None):
    start_time = datetime.now()
    print(f"\n{model_name} 开始训练时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    history = {
        'loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': [],
        'precision': [], 'recall': [], 'f1': [], 'roc_auc': [], 'epoch_time': []
    }

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in range(num_epochs):
        epoch_start = datetime.now()
        model.train()
        running_loss = 0.0
        
        for i in range(0, len(train_X), batch_size):
            inputs = train_X[i:i + batch_size]
            targets = train_y[i:i + batch_size]
            
            optimizer.zero_grad()
            outputs = model(inputs, feature_weights)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 验证集评估
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_X, feature_weights)
            val_loss = criterion(val_outputs, val_y)
            
            val_probs = val_outputs.cpu().numpy()
            val_preds = (val_outputs > 0.5).float().cpu().numpy()
            val_targets = val_y.cpu().numpy()

            val_acc = accuracy_score(val_targets, val_preds)
            val_pre = precision_score(val_targets, val_preds)
            val_rec = recall_score(val_targets, val_preds)
            val_f1 = f1_score(val_targets, val_preds)
            val_roc_auc = roc_auc_score(val_targets, val_probs)

        # 记录指标
        avg_train_loss = running_loss / (len(train_X) // batch_size)
        history['loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss.item())
        history['train_accuracy'].append(accuracy_score(
            train_y.cpu().numpy(), 
            (model(train_X, feature_weights) > 0.5).float().cpu().numpy()
        ))
        history['val_accuracy'].append(val_acc)
        history['precision'].append(val_pre)
        history['recall'].append(val_rec)
        history['f1'].append(val_f1)
        history['roc_auc'].append(val_roc_auc)
        history['epoch_time'].append((datetime.now() - epoch_start).total_seconds())

        scheduler.step(val_loss)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], 耗时: {history["epoch_time"][-1]:.2f}秒, '
            f'Train Loss: {history["loss"][-1]:.4f}, Train Accuracy: {history["train_accuracy"][-1]:.4f}, '
            f'Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_acc:.4f}'
        )

    # 测试集评估
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_X, feature_weights)
        test_loss = criterion(test_outputs, test_y)
        
        test_probs = test_outputs.cpu().numpy()
        test_preds = (test_outputs > 0.5).float().cpu().numpy()
        test_targets = test_y.cpu().numpy()

        test_acc = accuracy_score(test_targets, test_preds)
        test_pre = precision_score(test_targets, test_preds)
        test_rec = recall_score(test_targets, test_preds)
        test_f1 = f1_score(test_targets, test_preds)
        test_roc_auc = roc_auc_score(test_targets, test_probs)

    print(f"\n测试集结果: Loss: {test_loss.item():.4f}, Accuracy: {test_acc:.4f}, "
          f"F1: {test_f1:.4f}, ROC AUC: {test_roc_auc:.4f}")

    # 保存训练历史
    metrics_df = pd.DataFrame(history)
    os.makedirs(save_dir, exist_ok=True)
    metrics_df.to_csv(os.path.join(save_dir, f"{model_name}_metrics_{timestamp}.csv"), index=False)
    
    return history, (datetime.now() - start_time).total_seconds(), {
        'loss': test_loss.item(),
        'accuracy': test_acc,
        'precision': test_pre,
        'recall': test_rec,
        'f1': test_f1,
        'roc_auc': test_roc_auc
    }


# -------------------------- 敏感性分析函数 --------------------------
def analyze_single_feature(model_class, base_metrics, weights_range, feature_idx, feature_name, 
                          train_X, train_y, val_X, val_y, test_X, test_y, save_dir, params):
    """对单个特征进行敏感性分析，调整其权重并评估模型性能变化"""
    print(f"\n===== 开始对特征 '{feature_name}' 进行敏感性分析 =====")
    
    # 为该特征创建单独的结果目录
    feature_dir = os.path.join(save_dir, feature_name)
    os.makedirs(feature_dir, exist_ok=True)
    
    # 存储不同权重下的性能指标
    results = {
        'weights': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'loss': []
    }
    
    # 记录基线性能
    results['weights'].append(1.0)
    results['accuracy'].append(base_metrics['accuracy'])
    results['precision'].append(base_metrics['precision'])
    results['recall'].append(base_metrics['recall'])
    results['f1'].append(base_metrics['f1'])
    results['roc_auc'].append(base_metrics['roc_auc'])
    results['loss'].append(base_metrics['loss'])
    
    # 对每个权重值进行评估
    for weight in weights_range:
        if weight == 1.0:  # 跳过基线权重，已记录
            continue
            
        print(f"\n----- 评估特征权重: {weight} -----")
        
        # 创建特征权重向量（只调整目标特征，其他保持1.0）
        feature_weights = torch.ones(params['input_size'], dtype=torch.float32)
        feature_weights[feature_idx] = weight
        
        # 重新初始化模型
        model = model_class(
            conv_input=params['conv_input'], 
            input_size=params['input_size'], 
            hidden_size=params['hidden_size'], 
            num_layers=params['num_layers'], 
            num_heads=params['num_heads'], 
            dropout_prob=params['dropout_prob'], 
            output_size=params['output_size']
        ).to(device)
        
        # 训练并评估模型
        _, _, test_metrics = train_and_evaluate(
            model=model,
            model_name=f'CNN_LSTM_Attn_{feature_name}_weight_{weight}',
            criterion=params['criterion'],
            num_epochs=params['num_epochs'],
            batch_size=params['batch_size'],
            train_X=train_X,
            train_y=train_y,
            val_X=val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y,
            save_dir=feature_dir,
            timestamp=params['timestamp'],
            feature_weights=feature_weights
        )
        
        # 记录结果
        results['weights'].append(weight)
        results['accuracy'].append(test_metrics['accuracy'])
        results['precision'].append(test_metrics['precision'])
        results['recall'].append(test_metrics['recall'])
        results['f1'].append(test_metrics['f1'])
        results['roc_auc'].append(test_metrics['roc_auc'])
        results['loss'].append(test_metrics['loss'])
        
        # 每完成一个权重的训练就保存一次结果，防止意外中断
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(feature_dir, f'sensitivity_results.csv'), index=False)
    
    # 生成可视化结果
    plot_sensitivity_results(results, feature_name, feature_dir)
    
    print(f"===== 特征 '{feature_name}' 的敏感性分析已完成 =====")
    return results


def plot_sensitivity_results(results, feature_name, save_dir):
    """绘制敏感性分析结果图"""
    plt.figure(figsize=(15, 10))
    
    # 按指标绘制子图
    metrics = [
        ('accuracy', '准确率', 'green'),
        ('precision', '精确率', 'blue'),
        ('recall', '召回率', 'orange'),
        ('f1', 'F1分数', 'purple'),
        ('roc_auc', 'ROC AUC', 'red'),
        ('loss', '损失值', 'gray')
    ]
    
    for i, (metric, label, color) in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        plt.plot(results['weights'], results[metric], marker='o', color=color, linewidth=2)
        plt.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='基线权重')
        plt.title(f'{feature_name} - {label}')
        plt.xlabel('特征权重')
        plt.ylabel(label)
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'sensitivity_plots.png'), dpi=300)
    plt.close()


# -------------------------- 主函数 --------------------------
def main():
    # 配置参数
    data_path = "skill_builder_data_corrected_collapsed.csv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("experiment_results", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    # 创建敏感性分析结果目录
    sensitivity_dir = os.path.join(save_dir, "sensitivity_analysis")
    os.makedirs(sensitivity_dir, exist_ok=True)
    
    num_epochs = 50
    batch_size = 32
    
    # 加载数据
    data = load_and_preprocess_data(data_path)
    features = data['features']
    feature_names = data['feature_names']
    
    # 模型参数
    model_params = {
        'conv_input': 32,
        'input_size': len(features),
        'hidden_size': 64,
        'num_layers': 2,
        'num_heads': 4,
        'dropout_prob': 0.5,
        'output_size': 1,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'criterion': nn.BCELoss(),
        'timestamp': timestamp
    }
    
    # 首先训练基线模型（所有特征权重为1.0）
    print("===== 训练基线模型（所有特征权重为1.0） =====")
    base_model = CNN_LSTM_MultiHeadAttention(
        conv_input=model_params['conv_input'], 
        input_size=model_params['input_size'], 
        hidden_size=model_params['hidden_size'], 
        num_layers=model_params['num_layers'], 
        num_heads=model_params['num_heads'], 
        dropout_prob=model_params['dropout_prob'], 
        output_size=model_params['output_size']
    ).to(device)
    
    base_history, base_duration, base_metrics = train_and_evaluate(
        model=base_model,
        model_name='CNN_LSTM_Attn_baseline',
        criterion=model_params['criterion'],
        num_epochs=num_epochs,
        batch_size=batch_size,
        train_X=data['train_X'],
        train_y=data['train_y'],
        val_X=data['val_X'],
        val_y=data['val_y'],
        test_X=data['test_X'],
        test_y=data['test_y'],
        save_dir=save_dir,
        timestamp=timestamp,
        feature_weights=None  # 不调整权重，使用默认值1.0
    )
    
    # 保存基线模型结果
    base_results_df = pd.DataFrame(base_history)
    base_results_df.to_csv(os.path.join(save_dir, 'baseline_training_history.csv'), index=False)
    
    # 定义权重范围（从0.0到2.0，步长0.2）
    weights_range = np.arange(0.0, 2.1, 0.2)
    
    # 让用户选择要分析的特征
    print("\n可用特征列表:")
    for i, feature in enumerate(features):
        print(f"{i+1}. {feature_names.get(feature, feature)} ({feature})")
    
    while True:
        try:
            choice = int(input("\n请选择要分析的特征编号 (1-5, 输入0结束): "))
            if choice == 0:
                print("分析结束")
                break
            if 1 <= choice <= len(features):
                feature_idx = choice - 1
                feature = features[feature_idx]
                feature_name = feature_names.get(feature, feature)
                
                # 分析选中的特征
                analyze_single_feature(
                    model_class=CNN_LSTM_MultiHeadAttention,
                    base_metrics=base_metrics,
                    weights_range=weights_range,
                    feature_idx=feature_idx,
                    feature_name=feature_name,
                    train_X=data['train_X'],
                    train_y=data['train_y'],
                    val_X=data['val_X'],
                    val_y=data['val_y'],
                    test_X=data['test_X'],
                    test_y=data['test_y'],
                    save_dir=sensitivity_dir,
                    params=model_params
                )
                print(f"特征 '{feature_name}' 的分析结果已保存至: {os.path.join(sensitivity_dir, feature_name)}")
            else:
                print(f"请输入1到{len(features)}之间的数字，或输入0结束")
        except ValueError:
            print("请输入有效的数字")
    
    print("\n所有已选择的特征分析完成！结果已保存至:", sensitivity_dir)


if __name__ == "__main__":
    main()
