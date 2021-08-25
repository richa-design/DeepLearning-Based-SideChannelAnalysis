import os
import sys
import h5py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

SCALE = True # True for mixed precision

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
	file_path = os.path.normpath(file_path)
	if os.path.exists(file_path) == False:
		print("Error: provided file path '%s' does not exist!" % file_path)
		sys.exit(-1)
	return

# def load_sca_model(model_file):
# 	check_file_exists(model_file)
# 	try:
#         model = load_model(model_file)
# 	except:
# 		print("Error: can't load Keras model file '%s'" % model_file)
# 		sys.exit(-1)
# 	return model

#### ASCAD helper to load profiling and attack data (traces and labels)
# Loads the profiling and attack datasets from the ASCAD
# database
def load_ascad(ascad_database_file, load_metadata=False):
	check_file_exists(ascad_database_file)
	# Open the ASCAD database HDF5 for reading
	try:
		in_file  = h5py.File(ascad_database_file, "r")
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
		return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])

EMBED = True
REPORT = False
MORELINEAR = True
NEWLIN = False
batch_size = 20

class LSTMNet(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(LSTMNet, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        if (EMBED):
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        else:
            self.lstm = nn.LSTM(700, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        if (NEWLIN):
            self.fc1 = nn.Linear(hidden_dim, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, 1024)
            self.fc4 = nn.Linear(1024, 1024)
            self.fc = nn.Linear(1024, output_size)
        elif (MORELINEAR):
            self.fc1 = nn.Linear(hidden_dim, 4096)
            self.fc2 = nn.Linear(4096, 4096)
            self.fc = nn.Linear(4096, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)
            
        # self.softmax = nn.Softmax()
        
    def forward(self, x, hidden):
        # x = x.long()
        # xt[:,:,2] = x[2]
        if (REPORT):
            print("input shape: {}".format(x.shape))
            print("input: {}".format(x))
        if (EMBED):
            embeds = self.embedding(x)
            if (REPORT):
                print("embeds shape: {}".format(embeds.shape))
                print("embeds: {}".format(embeds))
            lstm_out, hidden = self.lstm(embeds, hidden)
        else:
            xt = x.view(x.shape[0], 1, x.shape[1])
            # print("xt shape: {}".format(xt.shape))
            lstm_out, hidden = self.lstm(xt, hidden)

        if (REPORT):
            print("lstm_out: {}".format(lstm_out))
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        if (REPORT):
            print("lstm_out_contig: {}".format(lstm_out))
        
        out = self.dropout(lstm_out)
        if (REPORT):
            print("dropout: {}".format(out))
        
        
        if (NEWLIN):
            out = self.fc1(out) 
            out = self.fc2(out) 
            out = self.fc3(out) 
            out = self.fc4(out) 
            out = self.fc(out)
        elif (MORELINEAR):
            out = self.fc1(out)
            out = self.fc2(out)
            out = self.fc(out)
        else:
            out = self.fc(out)
            
        # out = self.softmax(out)
        if (REPORT):
            print("fc: {}".format(out))
        out = out.view(batch_size, -1)
        # out = out[:,-1]
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

#%% Training
if __name__ == "__main__":
    dataset_root_dir = "/home/vernam/dl-project/ASCAD/ATMEGA_AES_v1/"
    #default parameters values
    ascad_database = dataset_root_dir + "/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
    #load traces
    (X_profiling, Y_profiling), (X_attack, Y_attack) = load_ascad(ascad_database)



    X_train, X_val, y_train, y_val = train_test_split(X_profiling,Y_profiling, shuffle = True,test_size=0.20, random_state=33)

    train_data = ASCAD(X_train, y_train)
    val_data = ASCAD(X_val, y_val)
    test_data = ASCAD(X_attack, Y_attack)

    # load data sets
    train_loader = DataLoader(train_data, batch_size=batch_size,
                        shuffle=True, num_workers=10) 
    val_loader = DataLoader(val_data, batch_size=batch_size,
                        shuffle=False, num_workers=10) 

    vocab_size = 257
    output_size = 256
    embedding_dim = 400
    hidden_dim = 512
    n_layers = 2

    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")


    model = LSTMNet(vocab_size, output_size, embedding_dim, hidden_dim, n_layers).to(device)
    # model.to(device)

    lr=1e-3
    # criterion = nn.functional.cross_entropy()
    # criterion = nn.NLLLoss()   
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

    epochs = 50
    counter = 0
    print_every = 100
    clip = 5
    valid_loss_min = np.Inf

    torch.autograd.set_detect_anomaly(True)

    # epochs = 75
    # if is_cuda:
    #     device.empty_cache()
    #     device.memory_summary(device=None, abbreviated=False)

    ### training
    # train_model(X_profiling, Y_profiling, best_model, training_model, epochs, batch_size)

    train_losses_all = []
    val_losses_all = []

    # print(len(X_attack))
    model.train()
    if SCALE:
        scaler = GradScaler()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)


    for epoch in range(epochs):
        h = model.init_hidden(batch_size)

        total_loss_train = 0
        total_loss_val = 0

        for i_train, data in enumerate(train_loader, 0):
            # print("train {}".format(i))
            inputs = data['trace']
            labels = data['label']
            inputs = inputs.to(device)
            labels = labels.to(device)

            counter += 1
            h = tuple([e.data for e in h])
            # if torch.cuda.is_available():
            #     print(inputs)
                # inputs, labels = inputs.to(device), labels.to(device) 
            # inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            with autocast():
                output, h = model(inputs, h)
                # print(output.squeeze().shape)
                # print(output)
                # loss = criterion(output.squeeze(), labels.float())
                # loss = criterion(torch.log(output.squeeze()), labels)
                loss = nn.functional.cross_entropy(output.squeeze(), labels)
                total_loss_train += loss.item()


            if SCALE:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            # optimizer.step()

            if counter%print_every == 0:
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for i_val, data in enumerate(val_loader, 0):
                    inp = data['trace'].to(device)
                    lab = data['label'].to(device)

                    val_h = tuple([each.data for each in val_h])
                    # inp, lab = inp.to(device), lab.to(device)
                    out, val_h = model(inp, val_h)
                    # val_loss = criterion(torch.log_softmax(out.squeeze()).clamp(min=1e-4), lab)
                    val_loss = nn.functional.cross_entropy(out.squeeze(), lab)
                    val_losses.append(val_loss.item())
                    total_loss_val += val_loss.item()

                model.train()
                print("Epoch: {}/{}...".format(epoch+1, epochs),
                    "Step: {}...".format(counter),
                    "Loss: {:.6f}...".format(loss.item()),
                    "Val Loss: {:.6f}".format(np.mean(val_losses)))
                if np.mean(val_losses) <= valid_loss_min:
                    torch.save(model.state_dict(), '/home/vernam/dl-project/bert_sca/lstm_save/state_dict.pt')
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                    valid_loss_min = np.mean(val_losses)
                # avg_val_loss = total_loss_val / i_val

        train_losses_all.append(total_loss_train / i_train)
        val_losses_all.append(np.mean(val_losses))
        plt.plot(np.arange(epoch+1), np.array(train_losses_all), 'b', label='LSTM_Train_Loss')
        plt.plot(np.arange(epoch+1), np.array(val_losses_all), 'r', label='LSTM_Val_Loss')
        plt.savefig("/home/vernam/dl-project/bert_sca/lstm_save/lstm_loss.png")
        plt.close()
