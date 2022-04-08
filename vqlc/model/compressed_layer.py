
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class attention2DCNNLayer(torch.nn.Module): # 2D
    def __init__(self, embed_dim, in_channel, out_channel, kernel_size, bias):
        super().__init__()

        self.in_channels = in_channel  # CNN
        self.out_channels = out_channel  # CNN
        self.d = embed_dim  # the dimension of embedding vectors
        self.num_kernels = in_channel * out_channel  # the number of weight clusters
        self.kernel_size = kernel_size
        self.n = kernel_size * kernel_size 
        self.Q = torch.nn.Parameter(
            torch.randn([self.out_channels, self.in_channels, self.d]))  # embedding vectors, trainable, random
        torch.nn.init.normal_(self.Q, mean=0, std=0.5)

        if bias:
            self.bias = torch.nn.Parameter(torch.randn(self.out_channels))
        else:
            self.bias = None

    def forward(self, x, K, V):
        p = self.get_p(K, 1)
        output = self.attention(p, V)
        x = torch.nn.functional.conv2d(x, output, bias=self.bias, stride=(1, 1), padding=(1, 1))
        loss2 = self.get_loss2(p)
        return x, loss2

    def get_p(self, K, m):
        p_scores = torch.einsum('ijk, km -> ijm', self.Q, K)
        p_scores = m * p_scores
        p = torch.nn.functional.softmax(p_scores, dim=2)

        return p

    def hard_weighting(self, x, K, V, m):
        p = self.get_p(K, m)
        output = self.attention(p, V)
        x = torch.nn.functional.conv2d(x, output, bias=self.bias, stride=(1, 1), padding=(1, 1))
        loss2 = self.get_loss2(p)
        return x, loss2

    def attention(self, p, V):

        output_weights = torch.einsum('ijk, kmn -> ijmn', p, V)  # 64,3,16 x 16,3,3 ---- 64,3,3,3

        return output_weights

    def get_loss2(self, p):

        a = torch.max(p, dim=2)
        return torch.sum(torch.square(a.values - 1.))

    def get_loss_q(self):

        loss_q = torch.sum(self.Q.pow(2))

        return loss_q



class attention3DCNNLayer(torch.nn.Module):
    def __init__(self, embed_dim, cluster_size, channels, kernel_size, bias):
        super().__init__()

        self.in_channels = channels[0]  # CNN
        self.out_channels = channels[1]  # CNN
        self.d = embed_dim  # the dimension of embedding vectors
        self.num_kernels = channels[0] * channels[1] // cluster_size  # the number of weight clusters
        self.kernel_size = kernel_size
        self.n = kernel_size * kernel_size 
        self.Q = torch.nn.Parameter(
            torch.randn(self.num_kernels, self.d))
        torch.nn.init.normal_(self.Q, mean=0, std=0.5)

        self.in_channels = channels[0]
        self.out_channels = channels[1]
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(self.out_channels))

    def forward(self, x, K, V):
        p = self.get_p(K, 1)
        output = self.attention(p, V)
        x = torch.nn.functional.conv2d(x, output, bias=self.bias, stride=(1, 1), padding=(1, 1))
        loss2 = self.get_loss2(p)
        return x, loss2

    def get_p(self, K, m):
        p_scores = torch.einsum('ik, km -> im', self.Q, K) 
        p_scores = m * p_scores
        p = torch.nn.functional.softmax(p_scores, dim=1)  
        return p

    def hard_weighting(self, x, K, V, m):
        p = self.get_p(K, m)
        output = self.attention(p, V)
        x = torch.nn.functional.conv2d(x, output, bias=self.bias, stride=(1, 1), padding=(1, 1))
        loss2 = self.get_loss2(p)
        return x, loss2

    def attention(self, p, V):
        output_weights = torch.einsum('ik, kgmn -> igmn', p, V) 
        kernel_weights = output_weights.view((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))

        return kernel_weights

    def get_loss2(self, p):
        a = torch.max(p, dim=1)
        return torch.sum(torch.square(a.values - 1.))

    def get_loss_q(self):
        loss_q = torch.sum(self.Q.pow(2))

        return loss_q