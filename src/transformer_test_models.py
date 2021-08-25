import os.path
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import ast
import torch
import torch.nn.functional as Func
from positional_encodings import PositionalEncoding1D
import torch.nn as nn
from transformer_sca import ASCAD

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
class Transformer(nn.Module):
	def __init__(self, src_vocab, labels, d_model, N, heads):
		super().__init__()
		self.embed = nn.Embedding(src_vocab, d_model)
		self.p_enc_1d = PositionalEncoding1D(d_model)
		encoder_layer = nn.TransformerEncoderLayer(d_model, heads)
		self.encoder = nn.TransformerEncoder(encoder_layer, N)
		self.fc1 = nn.Linear(d_model, 4096)
		self.fc2 = nn.Linear(4096, 4096)
		#self.fc3 = nn.Linear(4096, 4096)
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
		#fc1_output = self.fc1(e_outputs_avg)
		#fc2_output = self.fc2(fc1_output)
		#fc3_output = self.fc3(fc2_output)
		#output = self.out(fc2_output)
		#output = self.out(e_outputs_avg)
		prob_output = self.soft(output)
		return prob_output


AES_Sbox = np.array([
			0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
			0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
			0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
			0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
			0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
			0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
			0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
			0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
			0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
			0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
			0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
			0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
			0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
			0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
			0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
			0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
			])

def check_file_exists(file_path):
	file_path = os.path.normpath(file_path)
	if os.path.exists(file_path) == False:
		print("Error: provided file path '%s' does not exist!" % file_path)
		sys.exit(-1)
	return

def load_sca_model(model_file,params,device):
	check_file_exists(model_file)
	#params = (head*d_model, n, head, batch_size)
	
	d_model = int(params['d_model'])
	heads = int(params['heads'])
	N = int(params['N'])
	src_vocab = 256
	labels = 256
	batch_size = int(params['batch_size'])
	model = Transformer(src_vocab, labels, d_model, N, heads).to(device)
	#try:
	model.load_state_dict(torch.load(model_file,map_location='cuda:0'),strict=True)
	#except :
#		print("Error: can't load Keras model file '%s'" % model_file)#
		#sys.exit(-1)
	return model

# Compute the rank of the real key for a give set of predictions
def rank(predictions, metadata, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba):
	# Compute the rank
	if len(last_key_bytes_proba) == 0:
		# If this is the first rank we compute, initialize all the estimates to zero
		key_bytes_proba = np.zeros(256)
	else:
		# This is not the first rank we compute: we optimize things by using the
		# previous computations to save time!
		key_bytes_proba = last_key_bytes_proba

	for p in range(0, max_trace_idx-min_trace_idx):
		# Go back from the class to the key byte. '2' is the index of the byte (third byte) of interest.
		plaintext = metadata[min_trace_idx + p]['plaintext'][2]
		for i in range(0, 256):
			# Our candidate key byte probability is the sum of the predictions logs
			proba = predictions[p][AES_Sbox[plaintext ^ i]]
			if proba != 0:
				key_bytes_proba[i] += np.log(proba)
			else:
				# We do not want an -inf here, put a very small epsilon
				# that correspondis to a power of our min non zero proba
				min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
				if len(min_proba_predictions) == 0:
					print("Error: got a prediction with only zeroes ... this should not happen!")
					sys.exit(-1)
				min_proba = min(min_proba_predictions)
				key_bytes_proba[i] += np.log(min_proba**2)
	# Now we find where our real key candidate lies in the estimation.
	# We do this by sorting our estimates and find the rank in the sorted array.
	sorted_proba = np.array(list(map(lambda a : key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
	real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
	return (real_key_rank, key_bytes_proba)

def full_ranks(model, dataset, metadata, min_trace_idx, max_trace_idx, rank_step,device):
	# Real key byte value that we will use. '2' is the index of the byte (third byte) of interest.
	real_key = metadata[0]['key'][2]
	# Check for overflow
	if max_trace_idx > dataset.shape[0]:
		print("Error: asked trace index %d overflows the total traces number %d" % (max_trace_idx, dataset.shape[0]))
		sys.exit(-1)
	# Get the input layer shape
	input_layer_shape = model.embed.weight.shape

	input_data = dataset[min_trace_idx:max_trace_idx, :].view(np.uint8)
	input_data = torch.tensor(input_data).int().to(device)
	batch_size = 20
	input_data = torch.reshape(input_data,(-1,batch_size,700))

	index = np.arange(min_trace_idx+rank_step, max_trace_idx, rank_step)
	f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
	#TODO Test all attack traces using mini batches

	key_bytes_proba = []
	idx = np.arange(min_trace_idx+rank_step, max_trace_idx, rank_step)
	predictions = model(list(enumerate(input_data))[0][1]).cpu().data.numpy()
	for n, data in enumerate(input_data):
		if n==0:
		  continue
		# Predict our probabilities
		preds = model(data)
		predictions = np.concatenate((predictions, preds.cpu().data.numpy()),0)

	for t, i in zip(idx, range(0, len(idx))):
		real_key_rank, key_bytes_proba = rank( predictions[t-rank_step:t], metadata, real_key, t-rank_step, t, key_bytes_proba)
		f_ranks[i] = [t - min_trace_idx, real_key_rank] #f_ranks[n*len(idx)+i] = [t - min_trace_idx, real_key_rank]
		#break
	return f_ranks


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

# Check a saved model against one of the ASCAD databases Attack traces
def check_model(model_file, params, ascad_database, num_traces=2000,device='cuda'):
	check_file_exists(model_file)
	check_file_exists(ascad_database)
	# Load profiling and attack data and metadata from the ASCAD database
	(_, _), (X_attack, _), (_, Metadata_attack) = load_ascad(ascad_database, load_metadata=True)
	# Load model
	model = load_sca_model(model_file, params, device)
	
	# We test the rank over traces of the Attack dataset, with a step of 10 traces
	ranks = full_ranks(model, X_attack, Metadata_attack, 0, num_traces, 10,device)
	# We plot the results
	x = [ranks[i][0] for i in range(0, ranks.shape[0])]
	y = [ranks[i][1] for i in range(0, ranks.shape[0])]
	plt.title('Performance of '+model_file+' against '+ascad_database)
	plt.xlabel('number of traces')
	plt.ylabel('rank')
	plt.grid(True)
	plt.plot(y)
	plt.savefig(model_file+'_rank.png',block=False)
	plt.figure()

def read_parameters_from_file(param_filename):
	#read parameters for the extract_traces function from given filename
	#TODO: sanity checks on parameters
	param_file = open(param_filename,"r")

	#FIXME: replace eval() by ast.linear_eval()
	my_parameters= eval(param_file.read())

	model_file = my_parameters["model_file"]
	ascad_database = my_parameters["ascad_database"]
	num_traces = my_parameters["num_traces"]
	return model_file, ascad_database, num_traces

if __name__ == "__main__":
	if len(sys.argv)!=2:
		#default parameters values
		#model_file='/home/vernam/dl-project/transformer_sca/save/05-05'#"/home/vernam/dl-project/ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/cnn_best_ascad_desync0_epochs75_classes256_batchsize200.h5"
		model_file='/home/vernam/dl-project/transformer_sca/save/05-09-relu'
		ascad_database="/home/vernam/dl-project/ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
		num_traces=10000
	else:
		#get parameters from user input
		model_file, ascad_database, num_traces = read_parameters_from_file(sys.argv[1])

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	for model in os.listdir(model_file):

		if not 'model' in model or 'png' in model or 'models' in model:
			continue
		name = model.replace('model_(','').replace(')_best.pt','').split(',')
		params = {'d_model': name[0],
					'heads': name[2],
					'N': name[1],
					'batch_size': name[3]}
		#check model
		model = os.path.join(model_file, model)
	
		if os.path.exists(model+'_rank.png'):
			continue
		
		print(model)
		check_model(model, params, ascad_database, num_traces,device)


