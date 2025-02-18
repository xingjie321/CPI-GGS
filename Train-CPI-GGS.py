import random
import pickle
import sys
import timeit
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv, global_add_pool

import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_max_pool, global_mean_pool
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score


class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self, n_output=10, num_features_xd=10,output_dim=128, dropout=0.05):
        super(CompoundProteinInteractionPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
   
        self.n_output = n_output
        self.conv1 = GATv2Conv(num_features_xd, num_features_xd, heads=5)
        self.conv2 = GCNConv(num_features_xd,num_features_xd*5)
        self.conv3 = GCNConv(num_features_xd*5, num_features_xd*5)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1600)
        self.fc_g2 = torch.nn.Linear(1600, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(128, 10)

        self.W_cnn = nn.ModuleList([nn.Conv2d(
                    in_channels=1, out_channels=1, kernel_size=2*window+1,
                    stride=1, padding=window) for _ in range(layer_cnn)])
        self.bilstm=nn.GRU(dim,5,1,dropout=0.1,bidirectional=True)
        
        
        self.attention = nn.MultiheadAttention(embed_dim=2*dim, num_heads=1, batch_first=True)
        self.fc_output = nn.Linear(2*dim, 2)
        
    def cnn_BiGRU(self, xs, layer):

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)

        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))

        xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        bilstms,_=self.bilstm(xs)
        
        bilstms = torch.squeeze(bilstms, 0)

        return torch.unsqueeze(torch.mean(bilstms, 0), 0)

    def forward(self, inputs):
        fingerprints, edge_index, words = inputs

        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        if edge_index.size(1) > 0:
          edge_index = edge_index.t()
        z = self.conv1(fingerprint_vectors, edge_index)
        z = self.relu(z)
        y = self.conv2(fingerprint_vectors, edge_index)
        y = self.relu(y)
        y = self.conv3(y, edge_index)
        y = self.relu(y)
        x = torch.cat((z, y), dim=1)
        x_max = torch.max(x, dim=0, keepdim=True)[0]
        x_mean = torch.mean(x, dim=0, keepdim=True)
        x = torch.cat([x_max, x_mean], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.linear(x)
   
        word_vectors = self.embed_word(words)
        protein_vector = self.cnn_BiGRU(word_vectors, layer_cnn)

        cat_vector = torch.cat((x, protein_vector), 1)
        x = cat_vector.unsqueeze(1) 
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.squeeze(1)
        interaction = self.fc_output(attn_output)
        
        return interaction

    def __call__(self, data, train=True):

        inputs, correct_interaction = data[:-1], data[-1]
        
        predicted_interaction = self.forward(inputs)

        if train:
            loss = F.cross_entropy(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for data in dataset:
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        #N = len(dataset)
        #print(N)
        T, Y, S = [], [], []
        for data in dataset:
            (correct_labels, predicted_labels,
             predicted_scores) = self.model(data, train=False)
            T.append(correct_labels)
            Y.append(predicted_labels)
            S.append(predicted_scores)
        AUC = roc_auc_score(T, S)
        PRC = average_precision_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        return AUC, PRC, precision, recall

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy' ,allow_pickle=True) ]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    """Hyperparameters."""
    (DATASET, radius, ngram, dim, window, layer_cnn, layer_output,
     lr, lr_decay, decay_interval, weight_decay, iteration,
     setting) = ['human', 2, 3, 10, 11, 3, 3, 1e-3, 0.5, 10, 1e-6,100,
     'DATASEThuman--radius2--ngram3--dim10--window11--layer_cnn3--layer_output3--lr1e-3--lr_decay0.5--decay_interval10--weight_decay1e-6--iteration100']
    (dim, window, layer_cnn, layer_output, decay_interval,
     iteration) = map(int, [dim, window, layer_cnn, layer_output,
                            decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])


    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')



    """Load preprocessed data."""
    dir_input = ('./dataset/' + DATASET+ '/' )
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    edge_indexs = load_tensor(dir_input + 'edge_indexs', torch.LongTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'word_dict.pickle')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(compounds, edge_indexs, proteins, interactions))
    dataset = shuffle_dataset(dataset, random.randint(0, 10000))
    dataset_train, dataset_ = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

    """Set a model."""
    torch.manual_seed(random.randint(0, 10000))
    model = CompoundProteinInteractionPrediction().to(device)
    trainer = Trainer(model)
    tester = Tester(model)

    """Output files."""
    file_AUCs = './model/1.txt'
    file_model ='./model/model.h5'
    AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\t'
            'AUC_test\tPRC_test\tPrecision_test\tRecall_test')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    """Start training."""
    print('Training............CPI-GGS')
    print(AUCs)
    start = timeit.default_timer()

    for epoch in range(1, iteration):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)
        AUC_dev = tester.test(dataset_dev)[0]
        AUC_test, PRC_test, precision_test, recall_test = tester.test(dataset_test)
               end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train, AUC_dev,AUC_test, PRC_test, precision_test, recall_test]
        #tester.save_AUCs(AUCs, file_AUCs)
        #tester.save_model(model, file_model)

        print('\t'.join(map(str, AUCs)))
