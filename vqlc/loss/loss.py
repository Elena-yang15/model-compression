class Loss(nn.Module):
    def __init__(self, total_kernels):
        super().__init__()
        self.total_kernels = total_kernels
        
    def forward(self, loss_p, predicts, target, trade_off_p):
        loss1 = torch.nn.functional.cross_entropy(predicts, target)
        loss2 = 1./self.total_kernels * loss_p
        loss = loss1 + trade_off_p * loss2

        return loss, loss1, loss2