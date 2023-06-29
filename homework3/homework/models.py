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
    def __init__(self, layers=None, n_input_channels=3, kernel_size=3, stride=1, img_size=64*16*16, label=6):
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
        #print(x.shape)
        x = self.c_layers(x)
        return x

class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        raise NotImplementedError('FCN.__init__')

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
        raise NotImplementedError('FCN.forward')


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
