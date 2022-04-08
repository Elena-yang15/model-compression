def set_template(args):
    if args.template.find('CIFAR100') >= 0:
        args.data_train = 'CIFAR100'
        args.data_test = 'CIFAR100'

    if args.template.find('SVHN') >= 0:
        args.data_train = 'SVHN'
        args.data_test = 'SVHN'
        
    if args.template.find('MNIST') >= 0:
        args.data_train = 'MNIST'
        args.data_test = 'MNIST'
        args.n_colors = 1

    if args.template.find('VGG') >= 0:
        args.base = 'VGG'
        args.base_p = 'VGG'
        args.weight_decay = 5e-4

    if args.template.find('ResNet') >= 0:
        args.base = 'ResNet'
        args.base_p = 'ResNet'
        args.weight_decay = 1e-4


    if args.template.find('DenseNet') >= 0:
        if args.template.find('BC') >= 0:
            args.bottleneck = True
            args.reduction = 0.5
        else:
            args.bottleneck = False
            args.reduction = 1
        args.base = 'DenseNet'
        args.base_p = 'DenseNet'
        args.weight_decay = 1e-4
        args.print_every = 50
        args.nesterov = True

    if args.linear > 1:
        args.batch_size *= args.linear
        args.lr *= args.linear
        if args.decay.find('warm') < 0:
            args.decay = 'warm' + args.decay

        args.print_every /= args.linear

