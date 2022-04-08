import argparse
import template

parser = argparse.ArgumentParser(description='DQLC')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='',
                    help='You can set various templates in template.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='disable CUDA training')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', default='./')
parser.add_argument('--data_train', default='CIFAR10',
                    help='train dataset name')
parser.add_argument('--data_test', default='CIFAR10',
                    help='test dataset name')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_flip', action='store_true',
                    help='disable flip augmentation')
parser.add_argument('--crop', type=int, default=1,
                    help='enables crop evaluation')

# Model specifications
parser.add_argument('--model', default='DenseNet',
                    help='model name')
parser.add_argument('--vgg_type', type=str, default='16',
                    help='VGG type')
parser.add_argument('--act', default='relu',
                    help='activation function')

parser.add_argument('--depth', type=int, default=40,
                    help='number of convolution modules')
parser.add_argument('--in_channels', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--k', type=int, default=12,
                    help='DenseNet grownth rate')
parser.add_argument('--reduction', type=float, default=1,
                    help='DenseNet reduction rate')
parser.add_argument('--bottleneck', action='store_true',
                    help='ResNet/DenseNet bottleneck')

parser.add_argument('--kernel_size', type=int, default=3,
                    help='kernel size')
parser.add_argument('--no_bias', action='store_true',
                    help='do not use bias term for conv layer')
parser.add_argument('--precision', default='single',
                    help='model and data precision')

parser.add_argument('--num_clusters', type=int, default=256,
                    help='codebook size')
parser.add_argument('--embe_dim', type=int, default=50,
                    help='dimension of embedding vectors')
parser.add_argument('--cluster_size', type=int, default=128,
                    help='the number of input channels in a filter block')
parser.add_argument('--dropout', type=int, default=1e-1,
                    help='dropout')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--resume', type=int, default=-1,
                    help='load the model from the specified epoch')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training')

# Optimization specifications
parser.add_argument('--linear', type=int, default=1,
                    help='linear scaling rule')
parser.add_argument('--lr', type=float, default=1e-1,
                    help='learning rate')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='learning rate decay factor')

parser.add_argument('--optimizer', type=str, default='SGD',
                    help='optimizer to use')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--nesterov', action='store_true',
                    help='enable nesterov momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM betas')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=5e-5,
                    help='weight decay parameter')

# Loss specifications
parser.add_argument('--total_kernels', default=1622016/128,
                    help='the number of kernels in each model')
parser.add_argument('--trade_off_p', default=0.1,
                    help='trade off between two losses')
parser.add_argument('--increasing_rate', default=0.01,
                    help='the increasing rate of the trade off hyperparameter')

# Log specifications
parser.add_argument('--dir_save', default='./experiment',
                    help='the directory used to save')
parser.add_argument('--save', default='training',
                    help='file name to save')
parser.add_argument('--load', default='',
                    help='file name to load')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')


args = parser.parse_args()
template.set_template(args)

