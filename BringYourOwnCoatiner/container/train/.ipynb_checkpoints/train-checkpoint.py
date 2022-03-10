import argparse
import json
import logging
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models
import torchvision.transforms as transforms
from sklearn.preprocessing import MinMaxScaler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100


def split_data(stock, lookback):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 

        data.append(data_raw[index: index + lookback])
        
    data = np.array(data);
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);

    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out


def model_fn(model_dir):
    """
    Load the model for inference
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model.load_state_dict(torch.load(model_dir + "/ownScriptModel.pth"))
    model.eval()
    return model.to(device)


def get_model():
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    return model

def predict_fn(input_data, model):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        return model(input_data)


def _save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "ownScriptModel.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.state_dict(),path)


def _train(args,x_train,y_train_lstm,num_epochs):
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    
    hist = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []
    logger.info("Start training")
    for t in range(num_epochs):
        y_train_pred = model(x_train)

        loss = criterion(y_train_pred, y_train_lstm)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    training_time = time.time()-start_time
    logger.info("Training time: {}".format(training_time))
    return _save_model(model, args.model_dir)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosts", type=json.loads, default=os.environ["SM_HOSTS"])
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])

    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()
    file = os.path.join(args.train, "stockprices.csv")

    filepath = 's3://sagemaker-ap-southeast-1-205124155157/data/stockprices.csv'
    data = pd.read_csv(file)
#     data = data.sort_values('Date')
    price = data[['Close']]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))
    lookback = 5 # choose sequence length

    x_train, y_train, x_test, y_test = split_data(price, lookback)

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
    y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

    _train(parser.parse_args(),x_train,y_train_lstm,args.epochs)