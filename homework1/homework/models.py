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


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()

        """
        Your code here
        """
        self.norm = torch.nn.Flatten()
        self.fch = torch.nn.Linear(3 * 64 * 64, 6)
        # raise NotImplementedError('LinearClassifier.__init__')


    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        x = self.norm(x)
        x = self.fch(x)
        return x
        #raise NotImplementedError('LinearClassifier.forward')


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()

        """
        Your code here
        """
        self.fc1 = torch.nn.Linear(12288, 256)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, 6)

        # raise NotImplementedError('MLPClassifier.__init__')

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        x = x.view(x.size(0), -1)  # Flatten the input images
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
        # raise NotImplementedError('MLPClassifier.forward')


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
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
