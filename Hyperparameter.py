import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            accuracy_score, confusion_matrix, classification_report,
                            roc_curve, auc)
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
import copy

# Ignore warnings
warnings.filterwarnings("ignore")
# Define random seed
seed = 105
# Set random seed for reproducibility
def set_seed(seed=seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
set_seed()
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------------- Data Preprocessing --------------------------
# Read data (using OULA dataset)
data_path = "oula_merged_with_labels.csv"
df = pd.read_csv(data_path, encoding='utf-8')

# Add dataset ratio control parameter - adjust the proportion of data used
dataset_ratio = 1  # Example: 0.5 means using 50% of the data, 1.0 means using all data
print(f"Will use {dataset_ratio*100}% of the dataset for training and evaluation")

# Randomly sample the dataset by ratio
if dataset_ratio < 1.0:
    # Stratified sampling to maintain class distribution
    df = df.groupby('final_result', group_keys=False).apply(
        lambda x: x.sample(frac=dataset_ratio, random_state=seed)
    )
    # Reset index
    df = df.reset_index(drop=True)

# Check basic data information
print("Basic data information:")
df.info()
print("First few rows of data:")
print(df.head())
print("\nfinal_result class distribution:")
print(df['final_result'].value_counts())

# Create directory to save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("output_folder", f"{timestamp}_ratio_{dataset_ratio}")
os.makedirs(save_dir, exist_ok=True)
# Create directory to save training histories
history_dir = os.path.join(save_dir, "training_histories")
os.makedirs(history_dir, exist_ok=True)

# Label encoding (multi-class: final_result)
label_encoder_final = LabelEncoder()
df['final_result_encoded'] = label_encoder_final.fit_transform(df['final_result'])
num_classes = len(label_encoder_final.classes_)
class_names = label_encoder_final.classes_
print(f"Multi-class mapping: {dict(zip(class_names, range(num_classes)))}")

# Feature list (extended student features)
features = ['id_assessment', 'id_student', 'date_submitted', 'score', 
            'num_of_prev_attempts', 'studied_credits']
# Check if feature columns exist
missing_columns = [col for col in features if col not in df.columns]
if missing_columns:
    print(f"Warning: The following column names were not found in the DataFrame: {missing_columns}")
else:
    # Data preprocessing
    df = df.dropna(subset=features + ['final_result_encoded'])
    
    # Categorical feature encoding
    label_encoders = {}
    for col in ['id_assessment', 'id_student', 'code_module', 'code_presentation',
                'gender', 'region', 'highest_education', 'age_band', 'disability']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Feature normalization
    scalers = {}
    for feature in features:
        scaler = MinMaxScaler()
        df[feature] = scaler.fit_transform(df[feature].values.reshape(-1, 1)).flatten()
        scalers[feature] = scaler
    
    # Sort by time
    df = df.sort_values(by=['id_student', 'code_module', 'date_submitted'])
    
    # Create sequence data (target variable changed to final_result_encoded)
    def split_data(data, time_step):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data.iloc[i:i + time_step][features].values)
            y_val = data.iloc[i + time_step]['final_result_encoded']  # Multi-class target
            y.append(y_val)
        return np.array(X), np.array(y)
    
    # Create sequence data
    time_step = 30
    dataX, datay = split_data(df, time_step=time_step)
    print(f"Shape of sampled data sequences: dataX {dataX.shape}, datay {datay.shape}")
    
    # Split into training, validation, and test sets
    train_size = int(len(dataX) * 0.7)
    val_size = int(len(dataX) * 0.15)
    test_size = len(dataX) - train_size - val_size
    
    trainX, train_y = dataX[:train_size], datay[:train_size]
    valX, val_y = dataX[train_size:train_size+val_size], datay[train_size:train_size+val_size]
    testX, test_y = dataX[train_size+val_size:], datay[train_size+val_size:]
    
    # Convert to PyTorch tensors
    trainX = torch.FloatTensor(trainX).to(device)
    train_y = torch.LongTensor(train_y).to(device)
    valX = torch.FloatTensor(valX).to(device)
    val_y = torch.LongTensor(val_y).to(device)
    testX = torch.FloatTensor(testX).to(device)
    test_y = torch.LongTensor(test_y).to(device)
    
    # Save dataset split results
    np.save(os.path.join(save_dir, 'trainX.npy'), trainX.cpu().numpy())
    np.save(os.path.join(save_dir, 'train_y.npy'), train_y.cpu().numpy())
    np.save(os.path.join(save_dir, 'valX.npy'), valX.cpu().numpy())
    np.save(os.path.join(save_dir, 'val_y.npy'), val_y.cpu().numpy())
    np.save(os.path.join(save_dir, 'testX.npy'), testX.cpu().numpy())
    np.save(os.path.join(save_dir, 'test_y.npy'), test_y.cpu().numpy())
    
    # -------------------------- Feature Correlation Analysis --------------------------
    print("\nGenerating feature correlation matrix...")
    corr_matrix = df[features + ['final_result_encoded']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_correlation_matrix.png'))
    plt.close()
    # Save correlation matrix data
    corr_matrix.to_csv(os.path.join(save_dir, 'feature_correlation_matrix.csv'))
    
    # -------------------------- Model Definitions --------------------------
    # Core model: CNN-LSTM-Attention model (adapted for multi-class)
    class CNN_LSTM_Attn(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                     kernel_size=3, dropout=0.5):
            super(CNN_LSTM_Attn, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            # CNN layers
            self.conv1 = nn.Conv1d(in_channels=input_size, 
                                  out_channels=hidden_size, 
                                  kernel_size=kernel_size)
            self.conv2 = nn.Conv1d(in_channels=hidden_size, 
                                  out_channels=hidden_size, 
                                  kernel_size=kernel_size)
            
            # LSTM layer
            self.lstm = nn.LSTM(input_size=hidden_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True,
                               dropout=dropout)
            
            # Attention layer
            self.attention = nn.Linear(hidden_size * 2, 1)
            
            # Fully connected layer (output dimension is number of classes)
            self.fc = nn.Linear(hidden_size * 2, num_classes)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            # x shape: (batch_size, time_step, input_size)
            batch_size = x.size(0)
            
            # CNN part
            x = x.permute(0, 2, 1)  # Convert to (batch_size, input_size, time_step)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.permute(0, 2, 1)  # Convert back to (batch_size, time_step, hidden_size)
            
            # LSTM part
            h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
            lstm_out, _ = self.lstm(x, (h0, c0))  # (batch_size, time_step, hidden_size*2)
            
            # Attention mechanism
            attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # (batch_size, time_step, 1)
            attn_output = torch.sum(attn_weights * lstm_out, dim=1)  # (batch_size, hidden_size*2)
            
            # Output layer
            out = self.dropout(attn_output)
            out = self.fc(out)  # Multi-class output
            return out
    
    # Comparison model 1: Simplified SAINT
    class SAINT_Like(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(SAINT_Like, self).__init__()
            self.embedding = nn.Linear(input_size, hidden_size)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dropout=0.3),
                num_layers=2
            )
            self.fc = nn.Linear(hidden_size, num_classes)
            
        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.fc(x)
    
    # Comparison model 2: Simplified BERT4KT
    class BERT4KT_Like(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(BERT4KT_Like, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size)
            )
            self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)
            self.fc = nn.Linear(hidden_size, num_classes)
            
        def forward(self, x):
            x = self.encoder(x)
            attn_output, _ = self.attention(x, x, x)
            x = x + attn_output  # Residual connection
            x = x.mean(dim=1)
            return self.fc(x)
    
    # Comparison model 3: Basic LSTM model
    class Basic_LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
            super(Basic_LSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, num_classes)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.dropout(out[:, -1, :])  # Take last time step
            out = self.fc(out)
            return out
    
    # Comparison model 4: Basic CNN model
    class Basic_CNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, kernel_size=3, dropout=0.5):
            super(Basic_CNN, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size)
            self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size*2, kernel_size=kernel_size)
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(hidden_size*2, num_classes)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            x = x.permute(0, 2, 1)  # (batch, features, seq_len)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x).squeeze(2)
            x = self.dropout(x)
            x = self.fc(x)
            return x
    
    # Comparison model 5: GRU model
    class GRU_Model(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
            super(GRU_Model, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, num_classes)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            out, _ = self.gru(x, h0)
            out = self.dropout(out[:, -1, :])
            out = self.fc(out)
            return out
    
    # Comparison model 6: Bidirectional LSTM model
    class BiLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
            super(BiLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                               bidirectional=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size*2, num_classes)  # Multiply by 2 for bidirectional
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.dropout(out[:, -1, :])
            out = self.fc(out)
            return out
    
    # Comparison model 7: Transformer model
    class Transformer_Model(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, nhead=2, num_layers=1, dropout=0.3):
            # Reduce nhead (number of attention heads) and num_layers
            super(Transformer_Model, self).__init__()
            self.embedding = nn.Linear(input_size, hidden_size//2)  # Reduce hidden layer size
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_size//2, nhead=nhead, dropout=dropout),
                num_layers=num_layers
            )
            self.fc = nn.Linear(hidden_size//2, num_classes)
    
    # Comparison model 8: CNN-LSTM hybrid model
    class CNN_LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, kernel_size=3, dropout=0.5):
            super(CNN_LSTM, self).__init__()
            self.conv = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, num_classes)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = F.relu(self.conv(x))
            x = x.permute(0, 2, 1)
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            out = self.fc(out)
            return out
    
    # Comparison model 9: Simple fully connected network
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, time_step=30, dropout=0.5):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size * time_step, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size//2)
            self.fc3 = nn.Linear(hidden_size//2, num_classes)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten sequence
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    # Comparison model 10: LSTM-Attention model
    class LSTM_Attn(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
            super(LSTM_Attn, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                               bidirectional=True, dropout=dropout)
            self.attention = nn.Linear(hidden_size*2, 1)
            self.fc = nn.Linear(hidden_size*2, num_classes)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
            out, _ = self.lstm(x, (h0, c0))  # (batch, seq_len, hidden*2)
            
            # Attention mechanism
            attn_weights = F.softmax(self.attention(out), dim=1)
            attn_out = torch.sum(attn_weights * out, dim=1)
            
            out = self.dropout(attn_out)
            out = self.fc(out)
            return out
    
    # -------------------------- Training and Evaluation Functions --------------------------
    # Modify train_model function to add batch_size parameter
    def train_model(model, trainX, train_y, valX, val_y, epochs=50, lr=0.001, batch_size=32):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0

            # Process training data in batches
            for i in range(0, len(trainX), batch_size):
                optimizer.zero_grad()

                # Get batch data
                batch_X = trainX[i:i+batch_size]
                batch_y = train_y[i:i+batch_size]

                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                # Calculate accuracy
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == batch_y).sum().item()
                total_samples += batch_y.size(0)
                total_loss += loss.item() * batch_y.size(0)

                # Backward pass
                loss.backward()
                optimizer.step()

            # Calculate average loss and accuracy
            train_loss = total_loss / total_samples
            train_acc = total_correct / total_samples

            # Evaluate on validation set (also using batch processing)
            model.eval()
            val_loss = 0
            val_correct = 0
            val_samples = 0

            with torch.no_grad():
                for i in range(0, len(valX), batch_size):
                    batch_X = valX[i:i+batch_size]
                    batch_y = val_y[i:i+batch_size]

                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == batch_y).sum().item()
                    val_samples += batch_y.size(0)
                    val_loss += loss.item() * batch_y.size(0)

            val_loss /= val_samples
            val_acc = val_correct / val_samples

            # Record metrics
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if (epoch+1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            scheduler.step(val_loss)

        return model, history
    
    def evaluate_model(model, testX, test_y, model_name, class_names, save_dir, num_classes):
        model.eval()
        with torch.no_grad():
            outputs = model(testX)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            _, preds = torch.max(outputs, 1)

            # Save prediction results and probabilities
            np.save(os.path.join(save_dir, f'{model_name}_preds.npy'), preds.cpu().numpy())
            np.save(os.path.join(save_dir, f'{model_name}_probs.npy'), probs)

            # Calculate basic metrics
            acc = accuracy_score(test_y.cpu(), preds.cpu())
            precision = precision_score(test_y.cpu(), preds.cpu(), average='macro')
            recall = recall_score(test_y.cpu(), preds.cpu(), average='macro')
            f1 = f1_score(test_y.cpu(), preds.cpu(), average='macro')

            print(f"\n{model_name} test set metrics:")
            print(f"Accuracy: {acc:.4f}")
            print(f"Precision (macro): {precision:.4f}")
            print(f"Recall (macro): {recall:.4f}")
            print(f"F1 score (macro): {f1:.4f}")

            # Detailed classification report
            print("\nClassification report:")
            report = classification_report(test_y.cpu(), preds.cpu(), target_names=class_names, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(os.path.join(save_dir, f'{model_name}_classification_report.csv'))
            print(classification_report(test_y.cpu(), preds.cpu(), target_names=class_names))

            # Confusion matrix
            cm = confusion_matrix(test_y.cpu(), preds.cpu())
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.title(f'{model_name} Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'))
            plt.close()
            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            cm_df.to_csv(os.path.join(save_dir, f'{model_name}_confusion_matrix.csv'))

            # Multi-class ROC curve (fixed version)
            plt.figure(figsize=(10, 8))
            ohe = OneHotEncoder(sparse=False)
            y_onehot = ohe.fit_transform(test_y.cpu().reshape(-1, 1))

            roc_data = {}
            # Create standard FPR array (fixed length of 1000)
            standard_fpr = np.linspace(0, 1, 1000)

            for i in range(num_classes):
                class_name = class_names[i]
                # Extract true labels and predicted probabilities for current class
                y_true = y_onehot[:, i]
                y_score = probs[:, i]

                # Handle special cases: all predicted probabilities are the same or no positive samples
                if len(np.unique(y_score)) <= 1 or np.sum(y_true) == 0:
                    print(f"Warning: Insufficient data for class {class_name}, using default ROC curve")
                    # Use diagonal as default ROC curve
                    tpr = standard_fpr.copy()
                    roc_auc = 0.5  # AUC value for random guess
                else:
                    # Calculate original ROC curve
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                    # Interpolate TPR to standard FPR length
                    tpr = np.interp(standard_fpr, fpr, tpr)
                    roc_auc = auc(standard_fpr, tpr)

                # Store data (ensure all arrays have the same length)
                roc_data[f'{class_name}_fpr'] = standard_fpr
                roc_data[f'{class_name}_tpr'] = tpr
                roc_data[f'{class_name}_auc'] = np.full_like(standard_fpr, roc_auc)  # Match standard length

            # Plot diagonal line (random guess)
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            # Plot ROC curves for all classes
            for i in range(num_classes):
                class_name = class_names[i]
                plt.plot(roc_data[f'{class_name}_fpr'], 
                         roc_data[f'{class_name}_tpr'], 
                         lw=2, 
                         label=f'{class_name} (AUC = {roc_data[f"{class_name}_auc"][0]:.3f})')

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} Multi-class ROC Curve')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{model_name}_roc_curve.png'))
            plt.close()

            # Save ROC data after ensuring all data have the same length
            try:
                roc_df = pd.DataFrame(roc_data)
                roc_df.to_csv(os.path.join(save_dir, f'{model_name}_roc_data.csv'), index=False)
            except ValueError as e:
                print(f"Error saving ROC data: {e}")
                # Print length of each array for debugging
                for key, value in roc_data.items():
                    print(f"{key} length: {len(value)}")

            return {
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

    
    # -------------------------- Model Training and Evaluation --------------------------
    # Model parameters
    input_size = len(features)
    hidden_size = 64
    num_layers = 2
    epochs = 100  # Train all epochs without early stopping
    lr = 0.001
    
    # Initialize all models (1 core model + 10 comparison models) - keep original naming
    models = {
        'SAINT_Like': SAINT_Like(input_size, hidden_size, num_classes).to(device),
        'BERT4KT_Like': BERT4KT_Like(input_size, hidden_size, num_classes).to(device),
        'Basic_LSTM': Basic_LSTM(input_size, hidden_size, num_layers, num_classes).to(device),
        'Basic_CNN': Basic_CNN(input_size, hidden_size, num_classes).to(device),
        'GRU_Model': GRU_Model(input_size, hidden_size, num_layers, num_classes).to(device),
        'BiLSTM': BiLSTM(input_size, hidden_size, num_layers, num_classes).to(device),
        'Transformer_Model': Transformer_Model(input_size, hidden_size, num_classes).to(device),
        'CNN_LSTM': CNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device),
        'MLP': MLP(input_size, hidden_size, num_classes, time_step=time_step).to(device),
        'LSTM_Attn': LSTM_Attn(input_size, hidden_size, num_layers, num_classes).to(device),
        'CNN_LSTM_Attn': CNN_LSTM_Attn(input_size, hidden_size, num_layers, num_classes).to(device)
    }
    
    # Train all models
    results = {}
    histories = {}
    
    for name, model in models.items():
        print(f"\n----- Training {name} -----")
        trained_model, history = train_model(
            model, trainX, train_y, valX, val_y,
            epochs=epochs, lr=lr
        )
        histories[name] = history
        
        # Save complete training history
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(history_dir, f'{name}_training_history.csv'), index=False)
        
        # Evaluate model
        results[name] = evaluate_model(
            trained_model, testX, test_y, name, class_names, save_dir, num_classes 
        )
        
        # Save model
        torch.save(trained_model.state_dict(), os.path.join(save_dir, f'{name}_model.pth'))
        print(f"{name} model saved to {save_dir}")
    
    # -------------------------- Feature Importance Analysis --------------------------
    print("\nPerforming feature importance analysis...")
    # Use the trained core model for feature importance analysis
    feature_importance = []
    base_model = models['CNN_LSTM_Attn']
    # Load trained model parameters
    base_model.load_state_dict(torch.load(os.path.join(save_dir, 'CNN_LSTM_Attn_model.pth')))
    base_model.eval()
    
    # Base performance
    with torch.no_grad():
        base_outputs = base_model(testX)
        _, base_preds = torch.max(base_outputs, 1)
        base_f1 = f1_score(test_y.cpu(), base_preds.cpu(), average='macro')
    
    # Evaluate performance change by removing each feature individually
    for feature_idx, feature_name in enumerate(features):
        # Create test set with current feature removed
        modified_testX = testX.clone()
        modified_testX[:, :, feature_idx] = 0  # Set feature values to 0
        
        with torch.no_grad():
            outputs = base_model(modified_testX)
            _, preds = torch.max(outputs, 1)
            f1 = f1_score(test_y.cpu(), preds.cpu(), average='macro')
        
        # Importance = base performance - performance after removing feature
        importance = base_f1 - f1
        feature_importance.append((feature_name, importance))
    
    # Sort and visualize
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    names, importances = zip(*feature_importance)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(importances), y=list(names))
    plt.xlabel('Feature Importance (F1 Score Reduction)')
    plt.title('Impact of Different Features on Core Model Performance')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
    plt.close()
    
    # Save feature importance results
    fi_df = pd.DataFrame({'Feature': names, 'Importance': importances})
    fi_df.to_csv(os.path.join(save_dir, 'feature_importance.csv'), index=False)
    print("Feature importance analysis completed, results saved")
    
    # -------------------------- Result Summary --------------------------
    # Save evaluation results of all models
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(save_dir, 'model_comparison_results.csv'))
    print("\nModel comparison results:")
    print(results_df)
    
    # Plot model performance comparison
    plt.figure(figsize=(14, 8))
    results_df.plot(kind='bar')
    plt.title('Multi-class Performance Comparison of All Models')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
    plt.close()
    
    print(f"\nAll results saved to: {save_dir}")
