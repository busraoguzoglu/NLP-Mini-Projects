import torch.nn as nn
import torch
import math

# To solve Intel related matplotlib/torch error.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Net(nn.Module):

    # Define the layers
    def __init__(self, vocabulary, sequence_length):
        super(Net, self).__init__()

        # Vocabulary definition
        self.vocabulary = vocabulary
        self.num_words: int = 2000
        self.seq_len = sequence_length
        self.embedding_size = 64

        # CNN parameters definition
        self.kernel_1 = 2
        self.kernel_2 = 3
        self.kernel_3 = 4

        # Output size for each convolution
        # number of output channels of the convolution for each layer
        self.out_size = 32

        # Number of strides for each convolution
        # number of jumps that will be considered when sliding the window (the kernel)
        self.stride = 1

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

    def calc_out_conv_features(self, kernel):
        '''Calculates the number of output features after Convolution + Max pooling

            Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
            Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1

            source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        '''
        out_conv = ((self.embedding_size - 1 * (kernel - 1) - 1) / self.stride) + 1
        out_conv = math.floor(out_conv)
        return out_conv

    def calc_out_pool_features(self, kernel, out_conv):
        '''Calculates the number of output features after Convolution + Max pooling

            Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
            Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1

            source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        '''
        out_pool = ((out_conv - 1 * (kernel - 1) - 1) / self.stride) + 1
        out_pool = math.floor(out_pool)
        return out_pool

    def in_features_fc(self):

        # Calculate size of convolved/pooled features for convolution layers
        out_conv_1 = self.calc_out_conv_features(self.kernel_1)
        out_pool_1 = self.calc_out_pool_features(self.kernel_1, out_conv_1)

        out_conv_2 = self.calc_out_conv_features(self.kernel_2)
        out_pool_2 = self.calc_out_pool_features(self.kernel_2, out_conv_2)

        out_conv_3 = self.calc_out_conv_features(self.kernel_3)
        out_pool_3 = self.calc_out_pool_features(self.kernel_3, out_conv_3)

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