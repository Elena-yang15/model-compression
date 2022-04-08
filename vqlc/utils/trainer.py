
class Trainer:
    def __init__(self, my_model, my_loss, args, loader, optimizer):
        self.model = my_model
        self loss = my_loss
        self.train_loader = loader.loader_train
        self.test_loader = loader.loader_test
        self.epoch = 0
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.optimizer = optimizer
        self.args = args
    
    def train(epoch):
        self.model.train()
        ave_loss = []
        ave_loss2 = []
        ave_loss1 = []
        ave_loss3 = []
        correct = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

        self.optimizer.zero_grad()
        output, loss_train, loss_t = self.model(data)
        # lossv = torch.sum(network.V.pow(2)) / 2
        # lossq = network.loss_q()

        loss, loss1, loss2, loss3 = self.loss(loss_train, loss_t, output, target, self.args.trade_off_p,
                                            self.args.trade_off_t)

        loss.backward()
        self.optimizer.step()
        ave_loss.append(loss.item())
        ave_loss1.append(loss1.item())
        ave_loss2.append(loss2.item())
        ave_loss3.append(loss3.item())

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

        if batch_idx % log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, \tLoss1: {:.6f}, \tLossp: {:.6f}, \tLosst: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(), loss1.item(), loss2.item(), loss3.item()))

        ave_all = sum(ave_loss) / len(ave_loss)
        ave1 = sum(ave_loss1) / len(ave_loss1)
        ave2 = sum(ave_loss2) / len(ave_loss2)
        ave3 = sum(ave_loss3) / len(ave_loss3)

        print("Soft training: Average loss:{},\tAverage loss1:{},\tAverage loss2:{},\tAverage losst:{}".format(
        ave_all, ave1, ave2, ave3))
        accuracy = 100. * correct / len(self.train_loader.dataset)
        print("epochs: {},Soft training Accuracy: {}/{} ({:.0f}%)".format(
            epoch, correct, len(self.train_loader.dataset), 100. * correct / len(self.train_loader.dataset)))

        return ave_all, ave1, ave2, accuracy

    def soft_test():
        self.model.eval()
        test_loss = 0
        correct = 0
        test_loss1 = 0
        test_loss2 = 0
        test_loss3 = 0

        print('Start testing:')
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, loss_test, loss_t_test = self.model(data)

                test_loss_all, test_loss_1, test_loss_2, test_loss_3 = self.loss(loss_test, loss_t_test, output, target,
                                                                           self.args.trade_off_p, self.args.trade_off_t)  #
               test_loss += test_loss_all.item()
               test_loss1 += test_loss_1.item()
               test_loss2 += test_loss_2.item()
               test_loss3 += test_loss_3.item()

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

        test_loss /= len(self.test_loader.dataset) / self.args.batch_size_test
        test_loss1 /= len(self.test_loader.dataset) / self.args.batch_size_test
        test_loss2 /= len(self.test_loader.dataset) / self.args.batch_size_test
        test_loss3 /= len(self.test_loader.dataset) / self.args.batch_size_test

        print('\nTest set: Soft: loss: {:.4f},loss1: {:.4f},loss2: {:.4f}, losst: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, test_loss1, test_loss2, test_loss3, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

        return test_loss, test_loss1, test_loss2, 100. * correct / len(self.test_loader.dataset)
    
    def hard_test():
        self.model.eval()
        test_loss = 0
        correct = 0
        test_loss2 = 0
        print('Hard testing: Start testing:')
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, loss2 = self.model.hard_test(data)
                test_loss_all = torch.nn.functional.cross_entropy(output, target)
                test_loss += test_loss_all.item() 
                test_loss2 += loss2.item() 

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

        test_loss /= len(self.test_loader.dataset) / self.args.batch_size_test 
        test_loss2 /= len(self.test_loader.dataset) / self.args.batch_size_test 
        print('\nTest set: Hard: loss: {:.4f}, loss2: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, test_loss2, correct, len(self.test_loader.dataset),
        100. * correct / len(self.test_loader.dataset)))
    
        return test_loss, 100. * correct / len(self.test_loader.dataset)
   