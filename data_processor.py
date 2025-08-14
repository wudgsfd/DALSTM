import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def load_data(data_path="merged_student_assessments.csv"):
    df = pd.read_csv(data_path, encoding='GB2312')
    print("Basic data information:")
    df.info()
    print("First few rows of data:")
    print(df.head())
    return df


def preprocess_data(df, features=['id_assessment', 'id_student', 'date_submitted', 'score']):
    missing_columns = [col for col in features if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The following columns were not found in the DataFrame: {missing_columns}")

    df = df.dropna(subset=features)

    label_encoders = {}
    for col in ['id_assessment', 'id_student', 'code_module']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    scalers = {}
    for feature in features:
        scaler = MinMaxScaler()
        df[feature] = scaler.fit_transform(df[feature].values.reshape(-1, 1)).flatten()
        scalers[feature] = scaler

    df = df.sort_values(by=['id_student', 'code_module', 'date_submitted'])

    return df, label_encoders, scalers


def create_sequences(df, features, time_step=30, data_percentage=0.3):
    def split_data(data, time_step):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data.iloc[i:i + time_step][features].values)
            y_val = data.iloc[i + time_step]['code_module']
            y.append(y_val)
        return np.array(X), np.array(y)

    dataX, datay = split_data(df, time_step=time_step)
    print(f"Original shape of dataX: {dataX.shape}, number of features in last dimension: {dataX.shape[2]}")

    if dataX.shape[2] != 4:
        print(f"Warning: Feature dimension should be 4, actual is {dataX.shape[2]}, automatically corrected")
    dataX = dataX[:, :, :4]

    print(f"Corrected dataX shape: {dataX.shape}, number of features in last dimension: {dataX.shape[2]}")

    num_samples = int(dataX.shape[0] * data_percentage)
    dataX = dataX[:num_samples]
    datay = datay[:num_samples]
    print(f"Using {data_percentage * 100}% of the data, new dataX shape: {dataX.shape}, new datay shape: {datay.shape}")

    return dataX, datay


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
