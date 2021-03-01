import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from nltk import word_tokenize
import nltk
#nltk.download('punkt')
import math
from torch.utils.data import Dataset, DataLoader

# To solve Intel related matplotlib/torch error.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Some aux functions taken from:
# https://towardsdatascience.com/text-classification-with-cnns-in-pytorch-1113df31e79f

class Net(nn.Module):

    # Define the layers
    def __init__(self, vocabulary, sequence_length):
        super(Net, self).__init__()

        # Vocabulary definition
        self.vocabulary = vocabulary
        self.num_words: int = 1000
        self.seq_len = sequence_length
        self.embedding_size = 64

        # Dropout definition
        # self.dropout = nn.Dropout(0.2)

        # CNN parameters definition
        # Kernel sizes
        self.kernel_1 = 2
        self.kernel_2 = 3
        self.kernel_3 = 4

        # Output size for each convolution
        # number of output channels of the convolution for each layer
        self.out_size = 32

        # Number of strides for each convolution
        # number of jumps that will be considered when sliding the window (the kernel)
        self.stride = 2

        # Embedding layer definition
        self.embedding = nn.Embedding(self.num_words + 1, self.embedding_size, padding_idx=0)

        # Convolution layers definition
        self.conv_1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)
        self.conv_2 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_2, self.stride)
        self.conv_3 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_3, self.stride)

        # Max pooling layers definition
        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)

        # Fully connected layer definition
        self.fc = nn.Linear(self.in_features_fc(), 1)

    def in_features_fc(self):
        '''Calculates the number of output features after Convolution + Max pooling

        Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
        Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1

        source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        '''
        # Calculate size of convolved/pooled features for convolution_1/max_pooling_1 features
        out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)

        # Calcualte size of convolved/pooled features for convolution_2/max_pooling_2 features
        out_conv_2 = ((self.embedding_size - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_conv_2 = math.floor(out_conv_2)
        out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_pool_2 = math.floor(out_pool_2)

        # Calcualte size of convolved/pooled features for convolution_3/max_pooling_3 features
        out_conv_3 = ((self.embedding_size - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_conv_3 = math.floor(out_conv_3)
        out_pool_3 = ((out_conv_3 - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_pool_3 = math.floor(out_pool_3)

        # Returns "flattened" vector (input for fully connected layer)
        return (out_pool_1 + out_pool_2 + out_pool_3) * self.out_size

    # Define what to do on forward propagation
    def forward(self, x):

        # 1. Embedding Layer
        x = self.embedding(x)

        # 2. Conv layers
        x1 = self.conv_1(x)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)

        # Convolution layer 2 is applied
        x2 = self.conv_2(x)
        x2 = torch.relu(x2)
        x2 = self.pool_2(x2)

        # Convolution layer 3 is applied
        x3 = self.conv_3(x)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)

        # The output of each convolutional layer is concatenated into a unique vector
        union = torch.cat((x1, x2, x3), 2)
        union = union.reshape(union.size(0), -1)

        # The "flattened" vector is passed through a fully connected layer
        out = self.fc(union)
        # Dropout is applied
        m = nn.Dropout(p=0.2)
        out = m(out)
        # Activation function is applied
        out = torch.sigmoid(out)

        return out.squeeze()


class DatasetMapper(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_vocabulary(data, num_words):

    print('Building dictionary..')

    # Builds the vocabulary and keeps the "x" most frequent words
    vocabulary = dict()

    frequency_distribution = nltk.FreqDist()
    for sentence in data:
        for word in sentence:
            frequency_distribution[word] += 1

    common_words = frequency_distribution.most_common(num_words)

    for idx, word in enumerate(common_words):
        vocabulary[word[0]] = (idx + 1)

    print('Dictionary built.')
    return vocabulary


def word_to_idx(tokenized_data, vocabulary):
    # By using the dictionary (vocabulary), it is transformed
    # each token into its index based representation

    indexed_data = list()

    for sentence in tokenized_data:
        temp_sentence = list()
        for word in sentence:
            if word in vocabulary.keys():
                temp_sentence.append(vocabulary[word])
        indexed_data.append(temp_sentence)

    return indexed_data


def padding_sentences(indexed_data):
    # Each sentence which does not fulfill the required len
    # it's padded with the index 0

    # We need to first get max_sentence_length from our data
    max_sentence_length = len(indexed_data[0])
    for sentence in indexed_data:
        if len(sentence) > max_sentence_length:
            max_sentence_length = len(sentence)

    pad_idx = 0
    data_padded = list()

    for sentence in indexed_data:
        while len(sentence) < max_sentence_length:
            sentence.insert(len(sentence), pad_idx)
        data_padded.append(sentence)

    data_padded = np.array(data_padded)
    sequence_length = max_sentence_length

    return data_padded, sequence_length


def calculate_accuracy(grand_truth, predictions):
    true_positives = 0
    true_negatives = 0

    # Gets frequency  of true positives and true negatives
    # The threshold is 0.5
    for true, pred in zip(grand_truth, predictions):
        if (pred >= 0.5) and (true == 1):
            true_positives += 1
        elif (pred < 0.5) and (true == 0):
            true_negatives += 1
        else:
            pass
    # Return accuracy
    return (true_positives + true_negatives) / len(grand_truth)


def evaluation(model, loader_test):
    # Set the model in evaluation mode
    model.eval()
    predictions = []

    # Start evaluation phase
    with torch.no_grad():
        for x_batch, y_batch in loader_test:
            y_pred = model(x_batch)
            predictions += list(y_pred.detach().numpy())
    return predictions

def main():

    # Read data

    # columns: unnamed (id),title,text,label
    # [6335 x 4]
    news_data = pd.read_csv('data/RealFakeNews/news.csv')

    data = news_data['text'].values
    labels = news_data['label'].values
    classes = ('FAKE', 'REAL')

    converted_labels = labels.copy()

    for i in range(len(labels)):
        if labels[i] == 'FAKE':
            converted_labels[i] = 0
        else:
            converted_labels[i] = 1

    labels = converted_labels

    # Preprocessing

    # Convert to lowercase + clean symbols
    data = [x.lower() for x in data]
    data = [re.sub(r'[^A-Za-z]+', ' ', x) for x in data]

    # Tokenize (tokenized sentence by sentence)
    tokenized_data = [word_tokenize(x) for x in data]

    # Build dictionary (most common 500 words)
    vocabulary = build_vocabulary(tokenized_data, 1000)

    # Convert data to indexed form using dictionary
    # We get sentence by sentence tokenized/indexed data.
    indexed_data = word_to_idx(tokenized_data, vocabulary)

    # Padding: we need to make all sentence length equal.
    data_padded, sequence_length = padding_sentences(indexed_data)

    # Split train and test sets
    data_train, data_test, label_train, label_test = train_test_split(data_padded, labels,
                                                                      test_size=0.2, random_state=42)
    data = {'x_train': data_train, 'y_train': label_train, 'x_test': data_test, 'y_test': label_test}

    print(len(data_train))
    print(data['y_train'])

    # Training preparation: Initialize loaders
    # Initialize dataset maper
    train = DatasetMapper(data['x_train'], data['y_train'])
    test = DatasetMapper(data['x_test'], data['y_test'])

    # Initialize loaders
    train_loader = DataLoader(train, batch_size=40)
    test_loader = DataLoader(test, batch_size=40)

    # Initialize the network
    net = Net(vocabulary, sequence_length)

    # Define optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    epochs = 3

    # Starts training phase
    for epoch in range(epochs):

        running_loss = 0.0
        # Set model in training model
        net.train()

        predictions = []
        # Starts batch training
        for x_batch, y_batch in train_loader:

            print(len(x_batch))

            x_batch = x_batch.type(torch.LongTensor)
            y_batch = np.asarray(y_batch)
            y_batch = torch.tensor(y_batch)
            y_batch = y_batch.type(torch.FloatTensor)

            # Feed the model
            y_pred = net(x_batch)

            print(y_batch)
            print(y_pred)

            # Loss calculation
            loss = F.binary_cross_entropy(y_pred, y_batch)

            # zero the parameter gradients
            optimizer.zero_grad()

            # backward propagation and weight update
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Save predictions
            predictions += list(y_pred.detach().numpy())

        # Get test predictions
        test_predictions = evaluation(net, test_loader)

        # Evaluation
        train_accuracy = calculate_accuracy(label_train, predictions)
        test_accuracy = calculate_accuracy(label_test, test_predictions)
        print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (
        epoch + 1, loss.item(), train_accuracy, test_accuracy))

if __name__ == '__main__':
    main()