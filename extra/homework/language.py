import torch
from .models import LanguageModel, AdjacentLanguageModel, Bigram, load_model
# from models import LanguageModel, AdjacentLanguageModel, Bigram, load_model
from . import utils
# import utils
import numpy as np


def log_likelihood(model: LanguageModel, some_text: str):
    """
    Your code here

    Evaluate the log-likelihood of a given string.

    Hint: utils.one_hot might come in handy

    :param model: A LanguageModel
    :param some_text:
    :return: float
    """
    hot_text_encode = utils.one_hot(some_text)
    predicted_probs = model.predict_all(some_text)
    # print("p", predicted_probs[0])
    # adjusting predicted_probs shape to one hot tensor
    predicted_probs = predicted_probs[:, :predicted_probs.shape[1] - 1]
    # print(f'Encoded Text Shape {hot_text_encode.shape} log prob shap {predicted_probs.shape}')
    # getting the indexes where input string char is none zero
    indexes = torch.nonzero(hot_text_encode == 1)
    result_prob = predicted_probs[indexes[:, 0], indexes[:, 1]]
    return result_prob.sum()
    # raise NotImplementedError('log_likelihood') """


def sample_random(model: LanguageModel, max_length: int = 100):
    """
    Your code here.

    Sample a random sentence from the language model.
    Terminate once you reach a period '.'

    :param model: A LanguageModel
    :param max_length: The maximum sentence length
    :return: A string
    """

    # Initialize the sentence with the start token "Model"
    generated_sample = ''

    for _ in range(max_length):
        prediction_probs = model.predict_all(generated_sample)
        next_char_probabilities = prediction_probs[:, -1]
        next_char_probabilities = torch.exp(next_char_probabilities)  # Convert from log-probabilities
        next_char_probabilities /= next_char_probabilities.sum()  # Normalize probabilities to make them sum to 1

        # Sample the next character based on the probability distribution
        next_char_idx = torch.multinomial(next_char_probabilities, num_samples=1).item()
        next_char = utils.vocab[next_char_idx]

        generated_sample += next_char
        if next_char == '.':
            break

    return generated_sample


class TopNHeap:
    """
    A heap that keeps the top N elements around
    h = TopNHeap(2)
    h.add(1)
    h.add(2)
    h.add(3)
    h.add(0)
    print(h.elements)
    > [2,3]

    """

    def __init__(self, N):
        self.elements = []
        self.N = N

    def add(self, e):
        from heapq import heappush, heapreplace
        if len(self.elements) < self.N:
            heappush(self.elements, e)
        elif self.elements[0] < e:
            heapreplace(self.elements, e)


def beam_search(model: LanguageModel, beam_size: int, n_results: int = 10, max_length: int = 100,
                average_log_likelihood: bool = False):
    """
        Your code here

        Use beam search for find the highest likelihood generations, such that:
          * No two returned sentences are the same
          * the `log_likelihood` of each returned sentence is as large as possible

        :param model: A LanguageModel
        :param beam_size: The size of the beam in beam search (number of sentences to keep around)
        :param n_results: The number of results to return
        :param max_length: The maximum sentence length
        :param average_log_likelihood: Pick the best beams according to the average log-likelihood, not the sum
                                       This option favors longer strings.
        :return: A list of strings of size n_results
        """
    substring = ""
    beams_sum = list()
    beams_avg = list()
    array1 = list()
    # vocab = string.ascii_lowercase + ' .'
    if beam_size > 28:
        output1 = model.predict_all(substring)
        for i in range(len(output1)):
            char = utils.vocab[i]
            array1.append((char, output1[i]))
        beams_sum = array1
        beams_avg = array1
    loop_size = len(array1)

    array_period_avg = list()
    array_period_sum = list()
    for length in range(max_length):
        array_llsum = list()
        array_llavg = list()
        for k in range(loop_size):
            if average_log_likelihood:
                if beams_avg[k][0][-1] == '.':
                    array_period_avg.append((beams_avg[k][0], beams_avg[k][1]))
                    continue
                output = model.predict_all(beams_avg[k][0][-1])
            else:
                if beams_sum[k][0][-1] == '.':
                    array_period_sum.append((beams_sum[k][0], beams_sum[k][1]))
                    continue
                output = model.predict_all(beams_sum[k][0][-1])

            char_prob = output[:, -1]
            for i in range(len(char_prob)):
                char = utils.vocab[i]
                if not average_log_likelihood:
                    substring = beams_sum[k][0] + char
                    llsum = log_likelihood(model, substring)
                    array_llsum.append((substring, llsum))
                    if char == '.':
                        array_period_sum.append((substring, llsum))

                else:
                    substring = beams_avg[k][0] + char
                    llavg = log_likelihood(model, substring) / len(beams_avg[k][0])
                    array_llavg.append((substring, llavg))
                    if char == '.':
                        array_period_avg.append((substring, llavg))

        if not average_log_likelihood:
            seen = set()
            beams_sum = array_llsum
            beams_sum.extend(array_period_sum)
            beams_sum = [(a, b) for a, b in beams_sum if not (a in seen or seen.add(a))]
            beams_sum = sort_tuple(beams_sum)
            beams_sum = beams_sum[:beam_size]
            loop_size = len(beams_sum)

        else:
            seen = set()
            beams_avg = array_llavg
            beams_avg.extend(array_period_avg)
            beams_avg = [(a, b) for a, b in beams_avg if not (a in seen or seen.add(a))]
            beams_avg = sort_tuple(beams_avg)
            beams_avg = beams_avg[:beam_size]
            loop_size = len(beams_avg)

    if average_log_likelihood:
        seen = set()
        beams_avg.extend(array_period_avg)
        beams_avg = [(a, b) for a, b in beams_avg if not (a in seen or seen.add(a))]
        beams_avg = sort_tuple(beams_avg)
        substring = [i[0] for i in beams_avg]

    else:
        seen = set()
        beams_sum.extend(array_period_sum)
        beams_sum = [(a, b) for a, b in beams_sum if not (a in seen or seen.add(a))]
        beams_sum = sort_tuple(beams_sum)
        substring = [i[0] for i in beams_sum]

    return substring[:n_results]


# raise NotImplementedError('beam_search')


def sort_tuple(tup):
    return (sorted(tup, key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    """
      Some test code.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', choices=['Adjacent', 'Bigram', 'TCN'], default='Bigram')
    args = parser.parse_args()

    lm = AdjacentLanguageModel() if args.model == 'Adjacent' else (load_model() if args.model == 'TCN' else Bigram())

    ''' for s in ['abcdefg', 'abcgdef', 'abcbabc', '.abcdef', 'fedcba.']:
        print(s, float(log_likelihood(lm, s)))
    print()

    for i in range(10):
        s = sample_random(lm)
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()
    '''
    for s in beam_search(lm, 100):
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100, average_log_likelihood=True):
        print(s, float(log_likelihood(lm, s)) / len(s))
