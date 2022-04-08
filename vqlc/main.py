import torch
from data import Data
from model import Model
from loss import Loss
from option import args
from utils import Trainer
torch.manual_seed(args.seed)


loader = Data(args)
model = Model(args, checkpoint)
print(model)
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

loss = Loss(args.total_kernels)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) 
t = Trainer(model, loss, args, loader, optimizer)
for epoch in range(1, args.n_epochs + 1):
    t.train(epoch)
    t.soft_test()

t.hard_test()

    
