import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


# 读取数据
def load_data(data_path="merged_student_assessments.csv"):
    df = pd.read_csv(data_path, encoding='GB2312')
    print("数据基本信息：")
    df.info()
    print("数据的前几行：")
    print(df.head())
    return df


# 数据预处理
def preprocess_data(df, features=['id_assessment', 'id_student', 'date_submitted', 'score']):
    # 检查特征列是否存在
    missing_columns = [col for col in features if col not in df.columns]
    if missing_columns:
        raise ValueError(f"以下列名在 DataFrame 中找不到: {missing_columns}")

    # 移除缺失值
    df = df.dropna(subset=features)

    # 类别特征编码
    label_encoders = {}
    for col in ['id_assessment', 'id_student', 'code_module']:
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

    return df, label_encoders, scalers


# 创建序列数据
def create_sequences(df, features, time_step=30, data_percentage=0.3):
    def split_data(data, time_step):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data.iloc[i:i + time_step][features].values)
            y_val = data.iloc[i + time_step]['code_module']
            y.append(y_val)
        return np.array(X), np.array(y)

    dataX, datay = split_data(df, time_step=time_step)
    print(f"dataX 的原始形状: {dataX.shape}, 最后一维特征数: {dataX.shape[2]}")

    # 强制修正特征维度为4
    if dataX.shape[2] != 4:
        print(f"警告：特征维度应为4，实际为{dataX.shape[2]}，自动修正")
    dataX = dataX[:, :, :4]  # 只保留前4个特征

    print(f"修正后dataX形状: {dataX.shape}, 最后一维特征数: {dataX.shape[2]}")

    # 使用部分数据集
    num_samples = int(dataX.shape[0] * data_percentage)
    dataX = dataX[:num_samples]
    datay = datay[:num_samples]
    print(f"使用 {data_percentage * 100}% 的数据, 新数据X的形状: {dataX.shape}, 新数据y的形状: {datay.shape}")

    return dataX, datay


# 为Graph-based KT准备概念ID
def prepare_concept_ids(df, time_step, num_samples):
    concept_ids = []
    for i in range(len(df) - time_step):
        concept_seq = df.iloc[i:i + time_step]['id_assessment'].values
        if len(concept_seq) < time_step:
            concept_seq = np.pad(concept_seq, (0, time_step - len(concept_seq)), mode='constant')
        else:
            concept_seq = concept_seq[:time_step]
        concept_ids.append(concept_seq)
    concept_ids = np.array(concept_ids)[:num_samples]
    return concept_ids