import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv

torch.manual_seed(0)
np.random.seed(0)
scaler = MinMaxScaler(feature_range=(-1, 1))

# Input Steps
input_window = 65 
# Prediction Steps
output_window = 1 
batch_size = 10 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self,feature_size=250,num_layers=1,dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


# Example Input: [0..99]
# Example Target: [1..100]
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(inout_seq)
    

def create_pred_sequences(input_data):
    inout_seq = []
    inout_seq.append(input_data)
    return torch.FloatTensor(inout_seq)


def get_batch(source, i,batch_size):
    seq_len = min(batch_size, len(source) - i)
    data = source[i:i+seq_len]    
    # Feature Size = 1
    data_input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) 
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
    return data_input, target


def get_pred_batch(source, i,batch_size):
    seq_len = min(batch_size, len(source) - i)
    data = source[i:i+seq_len]    
    # Feature Size = 1
    data_input = torch.stack(torch.stack([item for item in data]).chunk(input_window,1)) 
    return data_input


def train(train_data):
    model.train()
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i,batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def plot_and_loss(eval_model, data_source,epoch):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source)):
            data, target = get_batch(data_source, i,1)
            output = eval_model(data)            
            total_loss += criterion(output, target).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
            
    len(test_result)

    pyplot.plot(scaler.inverse_transform(test_result.reshape(-1, 1)),color="red")
    pyplot.plot(scaler.inverse_transform(truth[:500].reshape(-1, 1)),color="blue")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    # pyplot.savefig('final%d.png'%epoch)
    pyplot.show()
    
    if (i == 0):
        return total_loss

    return total_loss / i


# Predict N Steps
def predict_future(eval_model, data_source,steps):
    eval_model.eval()
    data = get_pred_batch(data_source, 0,1)
    with torch.no_grad():
        for _ in range(0, steps):            
            output = eval_model(data[-input_window:])                        
            data = torch.cat((data, output[-1:]))
            
    data = data.cpu().view(-1)
    
    # Long Term Plot
    pyplot.plot(scaler.inverse_transform(data.reshape(-1, 1)),color="red")       
    pyplot.plot(scaler.inverse_transform(data[:input_window].reshape(-1, 1)),color="blue")    
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    # pyplot.savefig('final-future%d.png'%steps)
    pyplot.show()


def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size)
            output = eval_model(data)            
            total_loss += len(data[0])* criterion(output, targets).cpu().item()
    return total_loss / len(data_source)


# series = read_csv('Data/TSLA.csv', header=0, index_col=0, parse_dates=True, squeeze=True)["4. close"]
series = read_csv('Data/AAPL.csv', header=0, index_col=0, parse_dates=True, squeeze=True)["5. adjusted close"]
    
amplitude = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)

# samples = 1788
samples = 4200

train_data = amplitude[:samples]
test_data = amplitude[samples:]
final_data = amplitude[-input_window:]

final_data = create_pred_sequences(final_data)
final_data = final_data.to(device)

train_sequence = create_inout_sequences(train_data,input_window)
test_data = create_inout_sequences(test_data,input_window)

train_data = train_sequence.to(device)
val_data = test_data.to(device)


print(train_data.shape)
print(val_data.shape)
print(scaler.inverse_transform(amplitude.reshape(-1, 1)))


model = TransAm().to(device)

criterion = nn.MSELoss()
lr = 0.005 
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

epochs = 250

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)
    
    if(epoch % 250 is 0):
        val_loss = plot_and_loss(model, val_data,epoch)
    else:
        val_loss = evaluate(model, val_data)
   
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    scheduler.step() 

predict_future(model, final_data,65)