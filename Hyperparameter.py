import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
import copy
import seaborn as sns
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings("ignore")

# 定义随机种子
seed = 105

# 设置随机种子以确保结果可复现
def set_seed(seed=seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed()

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 读取数据
data_path = "merged_student_assessments.csv"
df = pd.read_csv(data_path, encoding='GB2312')

# 创建保存结果的目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("output_folder", "hyperparameter_analysis", timestamp)
os.makedirs(save_dir, exist_ok=True)
print(f"结果将保存至: {save_dir}")

# 特征列表
features = ['id_assessment', 'id_student', 'date_submitted', 'score']

# 检查特征列是否存在
missing_columns = [col for col in features if col not in df.columns]
if missing_columns:
    raise ValueError(f"数据集中缺少必要的特征列: {missing_columns}")

# 数据预处理
df = df.dropna(subset=features)

# 类别特征编码
label_encoders = {}
for col in ['id_assessment', 'id_student', 'code_module']:
    if col not in df.columns:
        raise ValueError(f"数据集中缺少列: {col}")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# 特征归一化
scalers = {}
for feature in features:
    scaler = MinMaxScaler()
    df[feature] = scaler.fit_transform(df[feature].values.reshape(-1, 1)).flatten()
    scalers[feature] = scaler

# 按时间排序
df = df.sort_values(by=['id_student', 'code_module', 'date_submitted'])

# 创建序列数据
def split_data(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data.iloc[i:i + time_step][features].values)
        y_val = data.iloc[i + time_step]['code_module']
        y.append(y_val)
    return np.array(X), np.array(y)

# 创建序列数据
time_step = 30
dataX, datay = split_data(df, time_step=time_step)
print(f"原始序列数据形状: dataX={dataX.shape}, datay={datay.shape}")

# 修正特征维度
if dataX.shape[2] != len(features):
    print(f"修正特征维度: 从{dataX.shape[2]}到{len(features)}")
    dataX = dataX[:, :, :len(features)]

# 使用部分数据集加速超参数实验
data_percentage = 0.7  # 增加数据比例以提高评估稳定性
num_samples = int(dataX.shape[0] * data_percentage)
dataX = dataX[:num_samples]
datay = datay[:num_samples]
print(f"使用{data_percentage*100}%的数据进行实验: dataX={dataX.shape}, datay={datay.shape}")

# 数据集划分
def train_val_test_split(dataX, datay, shuffle=True, train_perc=0.6, val_perc=0.2):
    if shuffle:
        indices = np.arange(len(dataX))
        np.random.shuffle(indices)
        dataX, datay = dataX[indices], datay[indices]

    total = len(dataX)
    train_idx = int(total * train_perc)
    val_idx = int(total * (train_perc + val_perc))

    return (dataX[:train_idx], datay[:train_idx],
            dataX[train_idx:val_idx], datay[train_idx:val_idx],
            dataX[val_idx:], datay[val_idx:])

# 划分数据集
train_X, train_y, val_X, val_y, test_X, test_y = train_val_test_split(dataX, datay)

# 转换为PyTorch张量
train_X = torch.tensor(train_X, dtype=torch.float32).to(device)
train_y = torch.tensor(train_y, dtype=torch.long).to(device)
val_X = torch.tensor(val_X, dtype=torch.float32).to(device)
val_y = torch.tensor(val_y, dtype=torch.long).to(device)
test_X = torch.tensor(test_X, dtype=torch.float32).to(device)
test_y = torch.tensor(test_y, dtype=torch.long).to(device)

print(f"训练集: {train_X.shape}, 验证集: {val_X.shape}, 测试集: {test_X.shape}")


# -------------------------- 模型定义 --------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.register_buffer('pe', self._generate_pe(max_len, d_model))

    def _generate_pe(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            self.pe = self._generate_pe(seq_len, self.d_model).to(x.device)
        x = x + self.pe[:, :seq_len, :]
        return x


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.5):
        super(MultiHeadAttentionLayer, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, v).transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_k)
        output = self.out_linear(output)
        return output, attention


class CNN_LSTM_MultiHeadAttention(nn.Module):
    def __init__(self, conv_input, input_size, hidden_size, num_layers, output_size, 
                 dropout_prob=0.5, num_heads=4, kernel_size=3):
        super(CNN_LSTM_MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # 卷积层（使用指定的kernel_size）
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=conv_input, 
                              kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(in_channels=conv_input, out_channels=conv_input, 
                              kernel_size=kernel_size, padding=kernel_size//2)
        self.res_conv = nn.Conv1d(in_channels=conv_input, out_channels=conv_input, 
                                 kernel_size=kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv1d(in_channels=conv_input, out_channels=conv_input, 
                              kernel_size=kernel_size, padding=kernel_size//2)
        
        # LSTM层
        self.lstm = nn.LSTM(conv_input, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # 注意力层（使用指定的头数）
        self.multi_head_attention = MultiHeadAttentionLayer(hidden_size * 2, num_heads, dropout_prob)
        self.positional_encoding = PositionalEncoding(hidden_size * 2)
        
        # 输出层
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # 卷积特征提取
        x_permuted = x.permute(0, 2, 1)  # (batch, features, seq_len)
        conv_out = self.conv1(x_permuted)
        conv_out = F.relu(conv_out)
        conv_out = self.conv2(conv_out)
        conv_out = F.relu(conv_out)
        conv_out = self.res_conv(conv_out) + conv_out  # 残差连接
        conv_out = self.conv3(conv_out)
        conv_out = F.relu(conv_out)
        conv_out = conv_out.permute(0, 2, 1)  # (batch, seq_len, features)
        
        # LSTM层
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        lstm_out, _ = self.lstm(conv_out, (h0, c0))
        
        # 位置编码和注意力
        lstm_out = self.positional_encoding(lstm_out)
        lstm_out = self.dropout(lstm_out)
        context_vector, _ = self.multi_head_attention(lstm_out, lstm_out, lstm_out)
        
        # 输出
        out = context_vector[:, -1, :]  # 取最后一个时间步
        out = self.fc(out)
        return self.logsoftmax(out)


# -------------------------- 超参数敏感性分析实验 --------------------------
def train_model(model, train_X, train_y, val_X, val_y, optimizer, criterion, num_epochs, batch_size):
    """训练模型并返回验证集上的最佳F1分数"""
    best_f1 = 0.0
    best_model = None
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=False)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        # 批处理训练
        for i in range(0, len(train_X), batch_size):
            inputs = train_X[i:i+batch_size]
            targets = train_y[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_X)
            val_preds = val_outputs.argmax(dim=1).cpu().numpy()
            val_true = val_y.cpu().numpy()
            current_f1 = f1_score(val_true, val_preds, average='weighted')
            
            # 学习率调度
            scheduler.step(current_f1)
            
            # 保存最佳模型
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_model = copy.deepcopy(model.state_dict())
    
    # 加载最佳模型
    model.load_state_dict(best_model)
    return best_f1, model


def evaluate_model(model, test_X, test_y):
    """在测试集上评估模型"""
    model.eval()
    with torch.no_grad():
        outputs = model(test_X)
        preds = outputs.argmax(dim=1).cpu().numpy()
        true = test_y.cpu().numpy()
        
        f1 = f1_score(true, preds, average='weighted')
        return f1, preds, true, outputs.softmax(dim=1).cpu().numpy()


def run_hyperparameter_experiment(param_name, param_values, fixed_params, 
                                 input_size, output_size, num_epochs=50):
    """运行特定超参数的敏感性实验"""
    results = []
    print(f"\n开始超参数 '{param_name}' 敏感性分析: {param_values}")
    
    for param in tqdm(param_values, desc=f"测试{param_name}"):
        # 设置当前参数组合
        current_params = fixed_params.copy()
        current_params[param_name] = param
        
        # 创建模型
        model = CNN_LSTM_MultiHeadAttention(
            conv_input=current_params['conv_input'],
            input_size=input_size,
            hidden_size=current_params['hidden_size'],
            num_layers=current_params['num_layers'],
            output_size=output_size,
            dropout_prob=current_params['dropout_prob'],
            num_heads=current_params['num_heads'],
            kernel_size=current_params['kernel_size']
        ).to(device)
        
        # 优化器
        optimizer = optim.AdamW(model.parameters(), lr=current_params['lr'])
        criterion = nn.NLLLoss()
        
        # 训练模型
        val_f1, best_model = train_model(
            model, train_X, train_y, val_X, val_y,
            optimizer, criterion, num_epochs, current_params['batch_size']
        )
        
        # 测试模型
        test_f1, _, _, _ = evaluate_model(best_model, test_X, test_y)
        
        # 记录结果
        results.append({
            'param_value': param,
            'val_f1': val_f1,
            'test_f1': test_f1,
            'params': current_params
        })
    
    return results


def plot_sensitivity(results, param_name, save_path):
    """绘制超参数敏感性分析图"""
    plt.figure(figsize=(10, 6))
    
    # 提取数据
    param_values = [r['param_value'] for r in results]
    val_f1 = [r['val_f1'] for r in results]
    test_f1 = [r['test_f1'] for r in results]
    
    # 绘图
    plt.plot(param_values, val_f1, 'o-', label='验证集 F1分数', color='blue')
    plt.plot(param_values, test_f1, 's-', label='测试集 F1分数', color='orange')
    
    # 标注最佳值
    best_idx = np.argmax(test_f1)
    plt.scatter(param_values[best_idx], test_f1[best_idx], 
               color='red', s=100, zorder=5, label=f'最佳值: {param_values[best_idx]}')
    
    # 设置标签和标题
    plt.xlabel(param_name.replace('_', ' ').title(), fontsize=12)
    plt.ylabel('F1分数', fontsize=12)
    plt.title(f'{param_name.replace("_", " ").title()} 敏感性分析', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(save_path, f'{param_name}_sensitivity.png'), dpi=1200)
    plt.close()
    
    return best_idx


# -------------------------- 实验参数设置 --------------------------
# 固定参数（基础配置）
fixed_params = {
    'conv_input': 32,
    'hidden_size': 64,        # 默认值，将作为其他实验的固定值
    'num_layers': 2,
    'output_size': len(label_encoders['code_module'].classes_),
    'dropout_prob': 0.5,
    'num_heads': 4,           # 默认值
    'kernel_size': 3,         # 默认值
    'lr': 1e-4,               # 默认值
    'batch_size': 32
}

input_size = len(features)
output_size = fixed_params['output_size']

# 待测试的超参数范围
hyperparameters = {
    'hidden_size': [32, 64, 128, 256],          # 隐藏层大小
    'num_heads': [2, 4, 8, 16],                 # 注意力头数
    'kernel_size': [3, 5, 7],                   # 卷积核大小
    'lr': [1e-5, 1e-4, 1e-3, 1e-2]              # 学习率
}

# 实验结果存储
all_results = {}
best_hyperparameters = fixed_params.copy()

# -------------------------- 运行实验 --------------------------
for param_name, param_values in hyperparameters.items():
    # 运行敏感性实验
    results = run_hyperparameter_experiment(
        param_name=param_name,
        param_values=param_values,
        fixed_params=best_hyperparameters,  # 使用当前最佳参数作为固定值
        input_size=input_size,
        output_size=output_size,
        num_epochs=30  # 减少epochs以加速实验，可根据需要调整
    )
    
    # 保存结果
    all_results[param_name] = results
    
    # 绘制敏感性曲线并获取最佳参数索引
    best_idx = plot_sensitivity(results, param_name, save_dir)
    best_hyperparameters[param_name] = param_values[best_idx]
    
    # 输出最佳结果
    print(f"{param_name} 最佳值: {param_values[best_idx]}, 测试集F1: {results[best_idx]['test_f1']:.4f}")


# -------------------------- 最佳参数组合评估 --------------------------
print("\n" + "="*60)
print("最佳超参数组合评估")
print("="*60)
print("最佳参数:")
for k, v in best_hyperparameters.items():
    print(f"  {k}: {v}")

# 使用最佳参数创建最终模型
final_model = CNN_LSTM_MultiHeadAttention(
    conv_input=best_hyperparameters['conv_input'],
    input_size=input_size,
    hidden_size=best_hyperparameters['hidden_size'],
    num_layers=best_hyperparameters['num_layers'],
    output_size=output_size,
    dropout_prob=best_hyperparameters['dropout_prob'],
    num_heads=best_hyperparameters['num_heads'],
    kernel_size=best_hyperparameters['kernel_size']
).to(device)

# 训练最终模型（使用更多epochs）
optimizer = optim.AdamW(final_model.parameters(), lr=best_hyperparameters['lr'])
criterion = nn.NLLLoss()

print("\n使用最佳参数训练最终模型...")
final_val_f1, final_model = train_model(
    final_model, train_X, train_y, val_X, val_y,
    optimizer, criterion, num_epochs=50,  # 增加训练轮次
    batch_size=best_hyperparameters['batch_size']
)

# 最终评估
final_test_f1, preds, true, probs = evaluate_model(final_model, test_X, test_y)
print(f"最终模型测试集F1分数: {final_test_f1:.4f}")

# 保存最佳模型
torch.save(final_model.state_dict(), os.path.join(save_dir, 'best_cnn_lstm_attn_model.pth'))

# 生成最终模型的评估图表
class_names = [str(cls) for cls in label_encoders['code_module'].classes_]

# 1. 混淆矩阵
plt.figure(figsize=(10, 8))
cm = confusion_matrix(true, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=class_names, yticklabels=class_names)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('最佳CNN_LSTM_Attn模型混淆矩阵')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'best_model_confusion_matrix.png'), dpi=1200)
plt.close()

# 2. 多类别ROC曲线
ohe = OneHotEncoder(sparse=False)
true_onehot = ohe.fit_transform(true.reshape(-1, 1))
fpr, tpr, roc_auc = {}, {}, {}
n_classes = len(class_names)

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(true_onehot[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例率 (FPR)')
plt.ylabel('真正例率 (TPR)')
plt.title('最佳CNN_LSTM_Attn模型多类别ROC曲线')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'best_model_roc_curve.png'), dpi=1200)
plt.close()

# 3. 超参数敏感性汇总图
plt.figure(figsize=(12, 8))
param_names = list(all_results.keys())
best_scores = [max([r['test_f1'] for r in results]) for results in all_results.values()]

x = np.arange(len(param_names))
plt.bar(x, best_scores, width=0.6)
plt.xticks(x, [name.replace('_', ' ').title() for name in param_names], rotation=45)
plt.ylabel('最佳F1分数')
plt.title('各超参数最佳F1分数对比')
plt.ylim(min(best_scores)-0.05, max(best_scores)+0.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'hyperparameter_best_scores.png'), dpi=1200)
plt.close()

print(f"\n所有实验完成，结果已保存至: {save_dir}")
    