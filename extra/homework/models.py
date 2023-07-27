import torch

from . import utils
from torch.nn.utils import weight_norm


class LanguageModel(object):
    def predict_all(self, some_text):
        """
        Given some_text, predict the likelihoods of the next character for each substring from 0..i
        The resulting tensor is one element longer than the input, as it contains probabilities for all sub-strings
        including the first empty string (probability of the first character)

        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: torch.Tensor((len(utils.vocab), len(some_text)+1)) of log-probabilities
        """
        raise NotImplementedError('Abstract function LanguageModel.predict_all')

    def predict_next(self, some_text):
        """
        Given some_text, predict the likelihood of the next character

        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: a Tensor (len(utils.vocab)) of log-probabilities
        """
        return self.predict_all(some_text)[:, -1]


class Bigram(LanguageModel):
    """
    Implements a simple Bigram model. You can use this to compare your TCN to.
    The bigram, simply counts the occurrence of consecutive characters in transition, and chooses more frequent
    transitions more often. See https://en.wikipedia.org/wiki/Bigram .
    Use this to debug your `language.py` functions.
    """

    def __init__(self):
        from os import path
        self.first, self.transition = torch.load(path.join(path.dirname(path.abspath(__file__)), 'bigram.th'))
        # print(" first.. ", self.first.shape)
        # print(" transition.. ", self.transition.shape)

    def predict_all(self, some_text):
        return torch.cat((self.first[:, None], self.transition.t().matmul(utils.one_hot(some_text))), dim=1)


class AdjacentLanguageModel(LanguageModel):
    """
    A simple language model that favours adjacent characters.
    The first character is chosen uniformly at random.
    Use this to debug your `language.py` functions.
    """

    def predict_all(self, some_text):
        prob = 1e-3 * torch.ones(len(utils.vocab), len(some_text) + 1)
        if len(some_text):
            one_hot = utils.one_hot(some_text)
            prob[-1, 1:] += 0.5 * one_hot[0]
            prob[:-1, 1:] += 0.5 * one_hot[1:]
            prob[0, 1:] += 0.5 * one_hot[-1]
            prob[1:, 1:] += 0.5 * one_hot[:-1]
        return (prob / prob.sum(dim=0, keepdim=True)).log()


def stack_param(param, input):
    stacks = []
    for i in range(input.shape[0]):
        stacks.append(param)
    batch = torch.stack(stacks, dim=0)
    return batch


class TCN(torch.nn.Module, LanguageModel):
    class CausalConv1dBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, dilation):
            """
            Your code here.
            Implement a Causal convolution followed by a non-linearity (e.g. ReLU).
            Optionally, repeat this pattern a few times and add in a residual block
            :param in_channels: Conv1d parameter
            :param out_channels: Conv1d parameter
            :param kernel_size: Conv1d parameter
            :param dilation: Conv1d parameter
            """
            super().__init__()
            self.c1 = torch.nn.ConstantPad1d((2 * dilation, 0), 0)
            self.c2 = weight_norm(torch.nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation))
            self.activation = torch.nn.ReLU()
            self.init_weights()

        def init_weights(self):
            self.c2.weight.data.normal_(0, 0.01)
            # raise NotImplementedError('CausalConv1dBlock.__init__')

        def forward(self, x):
            return self.activation(self.c2(self.c1(x)))
            # raise NotImplementedError('CausalConv1dBlock.forward')

    def __init__(self, layers=[32, 64, 128, 256]):
        """
        Your code here

        Hint: Try to use many layers small (channels <=50) layers instead of a few very large ones
        Hint: The probability of the first character should be a parameter
        use torch.nn.Parameter to explicitly create it.
        """
        super().__init__()
        c = len(utils.vocab)
        L = []
        total_dilation = 1
        for l in layers:
            L.append(self.CausalConv1dBlock(c, l, 3, total_dilation))
            total_dilation *= 2
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Conv1d(c, len(utils.vocab), 1)
        self.softmax = torch.nn.LogSoftmax(dim=1)

        self.first = torch.nn.Parameter(torch.rand(28, 1), requires_grad=True)
        # raise NotImplementedError('TCN.__init__')

    def forward(self, x):
        """
        Your code here
        Return the logit for the next character for prediction for any substring of x

        @x: torch.Tensor((B, vocab_size, L)) a batch of one-hot encodings
        @return torch.Tensor((B, vocab_size, L+1)) a batch of log-likelihoods or logits
        """
        if x.shape[2] == 0:
            return self.softmax(stack_param(self.first, x))

        output = self.classifier(self.network(x))
        # print("Model Output size ", output.shape)
        batch = stack_param(self.first, x)
        # batch = batch.squeeze(2)
        # print("Model batch size ", batch.shape)
        output = torch.cat([batch, output], dim=2)
        return output
        # raise NotImplementedError('TCN.forward')

    def predict_all(self, some_text):
        """
        Your code here

        @some_text: a string
        @return torch.Tensor((vocab_size, len(some_text)+1)) of log-likelihoods (not logits!)
        """
        one_hot = utils.one_hot(some_text)
        forward_output = self.forward(one_hot[None])
        prob = self.softmax(forward_output)
        forward_output = prob.view(one_hot.shape[0], one_hot.shape[1] + 1)
        return forward_output

        # raise NotImplementedError('TCN.predict_all')


def save_model(model):
    from os import path
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'tcn.th'))


def load_model():
    from os import path
    r = TCN()
    r.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th'), map_location='cpu'))
    return r