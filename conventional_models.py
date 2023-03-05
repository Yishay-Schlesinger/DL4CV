import torch


class TinyCNN(torch.nn.Module):
    """ 19,653,874 trainable params"""
    def __init__(self, n_classes=10):
        super(TinyCNN, self).__init__()

        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 280, kernel_size=7, padding=1, bias=False),
            torch.nn.BatchNorm2d(280),
            torch.nn.ELU(inplace=True),
        )
        kernel_size1 = 2 * int(round(3 * 0.66)) + 1
        self.pool1 = torch.nn.AvgPool2d(kernel_size=kernel_size1, stride=2)

        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(280, 560, kernel_size=5, padding=2, bias=False),
            torch.nn.BatchNorm2d(560),
            torch.nn.ELU(inplace=True)
        )

        kernel_size2 = 2 * int(round(3 * 0.66)) + 1
        self.pool2 = torch.nn.AvgPool2d(kernel_size=kernel_size2, stride=2)

        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(560, 1120, kernel_size=5, padding=2, bias=False),
            torch.nn.BatchNorm2d(1120),
            torch.nn.ELU(inplace=True)
        )
        # kernel_size3 = 2 * int(round(3 * 0.66)) + 1
        self.pool3 = torch.nn.AvgPool2d(kernel_size=3, stride=2)
        # number of output invariant channels
        c = 32
        self.invariant_map = torch.nn.Conv2d(1120, 32, kernel_size=1, bias=False)
        # Fully Connected classifier
        self.fully_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(c),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(c, n_classes),
        )

    def forward(self, input: torch.Tensor):
        """ input is expected to be a cube after image lifting and after masking"""
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = input
        # print(x.size())
        x = self.block1(x)  # block1
        # print(x.size())
        x = self.pool1(x)
        # print(x.size())
        x = self.block2(x)  #
        # print(x.size())
        x = self.pool2(x)
        # print(x.size())
        x = self.block3(x)
        # print(x.size())
        # pool over the spatial dimensions
        x = self.pool3(x)
        # print(x.size())
        # extract invariant features
        x = self.invariant_map(x)
        # print(x.size())
        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        # classify with the final fully connected layer
        x = self.fully_net(x.reshape(x.shape[0], -1))
        # print(x.size())
        return x


class TinyCNN2(torch.nn.Module):
    """ 19,653,874 trainable params"""
    def __init__(self, n_classes=10):
        super(TinyCNN2, self).__init__()

        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 280, kernel_size=7, padding=1, bias=False),
            torch.nn.BatchNorm2d(280),
            torch.nn.ELU(inplace=True),
        )
        kernel_size1 = 2 * int(round(3 * 0.66)) + 1
        self.pool1 = torch.nn.AvgPool2d(kernel_size=3, stride=2)

        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(280, 560, kernel_size=5, padding=2, bias=False),
            torch.nn.BatchNorm2d(560),
            torch.nn.ELU(inplace=True)
        )

        kernel_size2 = 2 * int(round(3 * 0.66)) + 1
        self.pool2 = torch.nn.AvgPool2d(kernel_size=3, stride=2)

        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(560, 1120, kernel_size=3, padding=0, bias=False),
            torch.nn.BatchNorm2d(1120),
            torch.nn.ELU(inplace=True)
        )
        # kernel_size3 = 2 * int(round(3 * 0.66)) + 1
        self.pool3 = torch.nn.AvgPool2d(kernel_size=3, stride=1)
        # number of output invariant channels
        c = 32
        self.invariant_map = torch.nn.Conv2d(1120, 32, kernel_size=1, bias=False)
        # Fully Connected classifier
        self.fully_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(c),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(c, n_classes),
        )

    def forward(self, input: torch.Tensor):
        """ input is expected to be a cube after image lifting and after masking"""
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = input
        # print(x.size())
        x = self.block1(x)  # block1
        # print(x.size())
        x = self.pool1(x)
        # print(x.size())
        x = self.block2(x)  #
        # print(x.size())
        x = self.pool2(x)
        # print(x.size())
        x = self.block3(x)
        # print(x.size())
        # pool over the spatial dimensions
        x = self.pool3(x)
        # print(x.size())
        # extract invariant features
        x = self.invariant_map(x)
        # print(x.size())
        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        # classify with the final fully connected layer
        x = self.fully_net(x.reshape(x.shape[0], -1))
        # print(x.size())
        return x