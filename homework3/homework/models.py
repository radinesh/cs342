import torch
import torch.nn.functional as F


class ClassificationLoss(torch.nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.ls = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        loss = self.ls(input, target)
        return loss
        # raise NotImplementedError('ClassificationLoss.forward')


class CNNClassifier(torch.nn.Module):
    def __init__(self, layers=None, n_input_channels=3, kernel_size=3, stride=1, img_size=64 * 16 * 16, label=6):
        super().__init__()
        """
        Your code here
        Hint: Base this on yours or HW2 master solution if you'd like.
        Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
        """
        if layers is None:
            layers = [32, 64]
        L = []
        C = []
        c = n_input_channels
        for l in layers:
            L.append(torch.nn.Conv2d(c, l, kernel_size, padding=(kernel_size - 1) // 2, stride=stride))
            L.append(torch.nn.ReLU())
            L.append(torch.nn.MaxPool2d(kernel_size=kernel_size - 1, stride=stride + 1))
            c = l
        hidden_layer_size = 128
        C.append(torch.nn.Linear(img_size, hidden_layer_size))
        C.append(torch.nn.ReLU())
        C.append(torch.nn.Linear(hidden_layer_size, label))
        self.f_layers = torch.nn.Sequential(*L)
        self.c_layers = torch.nn.Sequential(*C)

        # raise NotImplementedError('CNNClassifier.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        x = self.f_layers(x)
        x = x.view(x.size(0), -1)
        # print(x)
        # print(x.shape)
        x = self.c_layers(x)
        return x


class FCN(torch.nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        # Encoder
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # Decoder
        self.deconv4 = torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv1 = torch.nn.ConvTranspose2d(64, 5, kernel_size=4, stride=2, padding=1)

        # Skip connections
        self.skip1 = torch.nn.Conv2d(64, 64, kernel_size=1)
        self.skip2 = torch.nn.Conv2d(128, 128, kernel_size=1)
        self.skip3 = torch.nn.Conv2d(256, 256, kernel_size=1)

        # Activation function
        self.relu = torch.nn.ReLU()
        # raise NotImplementedError('FCN.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        # Decoder
        d4 = self.deconv4(x4)
        d3 = self.deconv3(d4)
        d2 = self.deconv2(d3)
        d1 = self.deconv1(d2)

        # Crop or pad the output to match the input size
        _, _, h, w = x.size()
        _, _, dh, dw = d1.size()

        if h != dh or w != dw:
            crop_h = (dh - h) // 2
            crop_w = (dw - w) // 2
            d1 = d1[:, :, crop_h:crop_h + h, crop_w:crop_w + w]

        return d1

        # raise NotImplementedError('FCN.forward')

    def crop_and_concat(self, x1, x2):
        diff_h = x2.size()[2] - x1.size()[2]
        diff_w = x2.size()[3] - x1.size()[3]

        # Crop x2 if necessary
        if diff_h != 0 or diff_w != 0:
            x2 = x2[:, :, :x1.size()[2], :x1.size()[3]]

        # Concatenate x1 and x2
        return torch.cat((x2, x1), dim=1)

model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
