import os
import sys
import time
import numpy as np
from torch._C import dtype
import h5py
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as Func
from positional_encodings import PositionalEncoding1D
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sklearn.model_selection import train_test_split
# Mixed precision makes it possible to train with large batch sizes
from torch.cuda.amp import GradScaler, autocast

from tensorboardX import SummaryWriter
#writer = SummaryWriter()
SCALE = True # True for mixed precision
torch.cuda.empty_cache()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#%% Data set class
class ASCAD():
    def __init__(self, X, Y) -> None:
        self.X = X
        self.Y = Y
        self.X = self.X.view(dtype=np.uint8)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        trace = torch.tensor(self.X[idx]).int()
        label = torch.tensor(self.Y[idx]).int()
        label = self.Y[idx].astype('long')
        sample = {'trace': trace, 'label': label}
        return sample


def check_file_exists(file_path):
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


def load_ascad(ascad_database_file, load_metadata=False):
    check_file_exists(ascad_database_file)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
        sys.exit(-1)
    # Load profiling traces
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
    # Load profiling labels
    Y_profiling = np.array(in_file['Profiling_traces/labels'])
    # Load attacking traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
    # Load attacking labels
    Y_attack = np.array(in_file['Attack_traces/labels'])
    if load_metadata == False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    else:
        return (X_profiling, Y_profiling), (X_attack, Y_attack), (
        in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])



class Transformer(nn.Module):
    def __init__(self, src_vocab, labels, d_model, N, heads):
        super().__init__()
        self.embed = nn.Embedding(src_vocab, d_model)
        self.p_enc_1d = PositionalEncoding1D(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, heads, dropout=0.5)
        self.encoder = nn.TransformerEncoder(encoder_layer, N)
        self.fc1 = nn.Linear(d_model, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.out = nn.Linear(4096, labels)
        #self.out = nn.Linear(d_model, labels)
        self.soft = nn.Softmax(dim=1)


    def forward(self, src, src_mask=None):
        embed_outputs = self.embed(src)
        p_outputs = self.p_enc_1d(embed_outputs)
        e_outputs = self.encoder(p_outputs, src_mask)
        # Average-pool over the number of features dimension and project to class probabilities
        e_outputs_avg = torch.mean(e_outputs, dim=1)
        fc1_output = self.fc1(e_outputs_avg)
        fc1_relu_output = Func.relu(fc1_output)
        fc2_output = self.fc2(fc1_relu_output)
        fc2_relu_output = Func.relu(fc2_output)
        output = self.out(fc2_relu_output)
        #output = self.out(e_outputs_avg)
        prob_output = self.soft(output)
        return output #prob_output


def train(device, transformer, train_loader, val_loader, params): #, X, Y, X_val, Y_val, epochs,batch_size = 200):
    if SCALE:
        scaler = GradScaler()
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    # this code is very important! It initialises the parameters with a
    # range of values that stops the signal fading or getting too big.
    # See this blog for a mathematical explanation.
    #optim = torch.optim.RMSprop(transformer.parameters(),lr=1e-4)
    optim = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    #optim = torch.optim.SGD(transformer.parameters(), 
    #                    lr=0.01,
    #                    momentum=0.9,
    #                    weight_decay=0.0003,
    #                    nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', verbose=True)
    start = time.time()
    #temp = start

    
    train_losses = []
    val_losses = []
    #Y_onehot = Y.detach().cpu().numpy()
    #Y_onehot = (np.arange(256) == Y_onehot[:, None]).astype(np.float32)
    #Y_onehot = torch.from_numpy(Y_onehot).to(device)

    for epoch in range(epochs):
        best_loss = 99
        for phase in ['train', 'val']:
            total_loss = 0
            if phase =='train':
                data_loader = train_loader
            else:
                data_loader = val_loader
            ctr = 0
            for i, data in enumerate(data_loader, 0): #for i in range(0, X.shape[0], batch_size):
                inputs = data['trace']
                labels = data['label']
                inputs = inputs.to(device)
                labels = labels.to(device)
                #src = X[i:i + batch_size, :]#.to(device)
                #target = Y_onehot[i:i + batch_size]
                #target = Y[i:i + batch_size].long()#.to(device)
                #fig = plt.figure()
                #plt.plot(inputs[25].cpu().data.numpy(),linewidth=1)
                #writer.add_figure('Input Trace',fig,0)
                #plt.savefig('trace_sample.png')
                #exit()
                optim.zero_grad()
                with autocast():
                    preds = transformer(inputs)
                    loss = Func.cross_entropy(preds, labels)
                if phase =='train':
                    if SCALE:
                        scaler.scale(loss).backward()
                        scaler.step(optim)
                        scaler.update()
                    else:
                        loss.backward()
                        optim.step() 
                          #total_loss += loss.data
                    if best_loss > loss:
                        #torch.save(transformer.state_dict(),"save/model_"+str(params)+"_best.pt")
                        torch.save(transformer.state_dict(),"model_"+str(params)+"_best.pt")
            
                total_loss += loss.item() #* inputs.size(0)
                
            #if epoch % 10 == 0:
            #    total_val_loss = 0
            #    for i in range(0, X_val.shape[0], batch_size):
            #        #target = Y_onehot[i:i + batch_size]
            #        src = X_val[i:i + batch_size, :]
            #        target = Y_val[i:i + batch_size].long()
            #        preds = transformer(src)
            #        loss = Func.cross_entropy(preds, target)
            #        total_val_loss += loss.data
            #    loss_val_avg = (total_val_loss*batch_size) / X.shape[0]
            loss_avg = total_loss / i #len(data_loader.dataset)
            #loss_avg = (total_loss*batch_size) / X.shape[0]
            print(phase," time = %dm, epoch %d, loss = %.3f" % ((time.time() - start) // 60,
                    epoch + 1, loss_avg))
            if phase == 'train':
                
                scheduler.step(loss_avg)
                train_losses.append(loss_avg)
                plt.plot(np.arange(epoch+1), np.array(train_losses), 'b', label='Transformer_Train_Loss')
            else:
                val_losses.append(loss_avg)
                plt.plot(np.arange(epoch+1), np.array(val_losses), 'r', label='Transformer_Val_Loss')

        #plt.savefig("save/transformer_loss"+str(params)+".png")
        plt.savefig("/home/vernam/dl-project/transformer_sca/save/transformer_loss"+str(params)+".png")
    plt.close()

if __name__ == "__main__":
    ascad_database = "/home/vernam/dl-project/ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
    # load traces
    (X_profiling, Y_profiling), (X_attack, Y_attack) = load_ascad(ascad_database)
    #X_profiling = X_profiling - np.min(X_profiling)
    #X_attack = X_attack - np.min(X_attack)

    X_train, X_val, y_train, y_val = train_test_split(X_profiling,Y_profiling, shuffle = True,test_size=0.20, random_state=33)

    db_train = ASCAD(X_train, y_train)
    db_val = ASCAD(X_val, y_val)
    # TODO Plot Full rank vs epoch
    db_test = ASCAD(X_attack, Y_attack)

    # parameters
    heads = [4] #[4,5,6]#[1,2,3,4]
    d_model = 128 #256 #128#16
    N = [4,5]#[1,2,3,4]
    epochs = 5
    src_vocab = np.max(X_profiling.view(np.uint8)) + 1
    labels = 256
    batch_sizes = [32]
    x = 0
    for head in heads:
        for n in N:
            for batch_size in batch_sizes:
                x = x +1
                if x < 0:
                    continue
                # load data sets
                train_loader = DataLoader(db_train, batch_size=batch_size,
                                    shuffle=True, num_workers=10) 
                val_loader = DataLoader(db_val, batch_size=batch_size,
                                    shuffle=False, num_workers=10) 
                #test_loader = DataLoader(db_train, batch_size=128,
                #                    shuffle=False, num_workers=10) 
                #print(X_profiling.shape)
                #print(Y_profiling.shape)
                #print(X_attack.shape)
                #print(Y_attack.shape)
                #print(np.max(X_profiling))
                #print(np.min(X_profiling))
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                #head = 4
                #n = 4
                #d_model = 128
                transformer = Transformer(src_vocab, labels, head*d_model, n, head).to(device)
         
                #transformer = nn.DataParallel(transformer)
                # X = torch.from_numpy(X_profiling).int().to(device)
                # Y = torch.from_numpy(Y_profiling).int().to(device)
                params = (head*d_model, n, head, batch_size)
                train(device, transformer, train_loader,val_loader, params) #, X, Y, X_val, Y_val, epochs,batch_size)
