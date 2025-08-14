import torch
import numpy as np
from config import set_seed, get_device, create_save_dir, train_val_test_split
from data_processor import load_data, preprocess_data, create_sequences, prepare_concept_ids
from models import (SAINT, RNNModel, GRUModel, LSTMModel, BiLSTMModel, TCN,
                    CNN_LSTM_MultiHeadAttention, GraphKT, BERT4KT, SAKT)
from trainer import train_and_evaluate


def main():

    set_seed()
    device = get_device()
    print(f"使用设备: {device}")


    save_dir, timestamp = create_save_dir()


    df = load_data("merged_student_assessments.csv")
    features = ['id_assessment', 'id_student', 'date_submitted', 'score']
    df, label_encoders, scalers = preprocess_data(df, features)


    time_step = 30
    data_percentage = 0.3
    dataX, datay = create_sequences(df, features, time_step, data_percentage)
    num_samples = dataX.shape[0]


    train_X, train_y, val_X, val_y, test_X, test_y = train_val_test_split(
        dataX, datay, shuffle=True, train_percentage=0.6, val_percentage=0.2, test_percentage=0.2)

    train_X, train_y = torch.tensor(train_X, dtype=torch.float32).to(device), torch.tensor(train_y,
                                                                                           dtype=torch.long).to(device)
    val_X, val_y = torch.tensor(val_X, dtype=torch.float32).to(device), torch.tensor(val_y, dtype=torch.long).to(device)
    test_X, test_y = torch.tensor(test_X, dtype=torch.float32).to(device), torch.tensor(test_y, dtype=torch.long).to(
        device)

    input_size = 4  
    hidden_size = 64  
    num_layers = 2  
    output_size = len(label_encoders['code_module'].classes_)  
    conv_input = 32  
    num_epochs = 100
    batch_size = 32
    max_seq_len = time_step  
    num_concepts = len(label_encoders['id_assessment'].classes_)  


    concept_ids = prepare_concept_ids(df, time_step, num_samples)
    concept_ids = torch.tensor(concept_ids, dtype=torch.long).to(device)

    saint_model = SAINT(
        dim_model=hidden_size,
        num_en=3,
        num_de=3,  
        heads_en=4,  
        total_ex=len(label_encoders['id_assessment'].classes_),  
        total_cat=len(label_encoders['id_student'].classes_),  
        total_in=2,  
        heads_de=4, 
        seq_len=time_step,
        output_size=output_size
    ).to(device)
    saint_model.set_vocab_sizes(
        total_ex=len(label_encoders['id_assessment'].classes_),
        total_cat=len(label_encoders['id_student'].classes_),
        total_in=2
    )

    models = {
        'SAINT': saint_model,
        'Graph-based KT': GraphKT(input_size, hidden_size, num_layers, output_size, num_concepts,
                                  concept_emb_size=input_size).to(device),
        'BERT4KT': BERT4KT(input_size, hidden_size, num_layers, output_size, max_seq_len).to(device),
        'SAKT': SAKT(input_size, hidden_size, output_size).to(device),
        'TCN': TCN(input_size, output_size, num_channels=[32, 64, 128], kernel_size=3, dropout=0.2).to(device),
        'RNN': RNNModel(input_size, hidden_size, num_layers, output_size).to(device),
        'GRU': GRUModel(input_size, hidden_size, num_layers, output_size).to(device),
        'LSTM': LSTMModel(input_size, hidden_size, num_layers, output_size).to(device),
        'BiLSTM': BiLSTMModel(input_size, hidden_size, num_layers, output_size).to(device),
        'CNN_LSTM_Attn': CNN_LSTM_MultiHeadAttention(conv_input, input_size, hidden_size, num_layers,
                                                     num_heads=4, dropout_prob=0.5, output_size=output_size).to(device),
    }


    criterion = torch.nn.CrossEntropyLoss()
    all_histories = {}
    all_predictions = {}

    for model_name, model in models.items():
        history, predictions, duration = train_and_evaluate(
            model, model_name, criterion, num_epochs, batch_size,
            train_X, train_y, val_X, val_y, test_X, test_y,
            save_dir, timestamp, device, concept_ids if model_name == 'Graph-based KT' else None
        )
        all_histories[model_name] = history
        all_predictions[model_name] = predictions

    print("\n所有模型训练完成！")


if __name__ == "__main__":

    main()
