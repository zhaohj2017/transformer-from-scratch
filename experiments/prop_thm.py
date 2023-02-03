from _context import former
from former import util, GTransformer
from util import d, here

import torch
from torch import nn
# from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as dist

import numpy as np

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import random, tqdm, sys, math, gzip

NUM_TOKENS = 128

# # Used for converting between nats and bits
# LOG2E = math.log2(math.e)

def sample(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome log-probabilities
    :param temperature: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """
    
    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs/temperature, dim=0)
    cd = dist.Categorical(p)

    return cd.sample() #discrete class, the index

def read_data(path, n_train=int(1e6)):
    """
    :param path:
    :param n_train:
    :param n_valid:
    :param n_test:
    :return:
    """
    with gzip.open(path) if path.endswith('.gz') else open(path) as file:
        X = np.frombuffer(file.read(n_train).encode(encoding='utf-8'), dtype=np.uint8)
        return torch.from_numpy(X.copy()), torch.from_numpy(X.copy()), torch.from_numpy(X.copy())

def go(arg):
    if arg.seed < 0:
        seed = random.randint(0, 1000000)
        print('random seed: ', seed)
    else:
        torch.manual_seed(arg.seed)

    tbw = SummaryWriter(log_dir=arg.tb_dir) # Tensorboard logging

    # set data path
    arg.data = here('data/test2.txt') if arg.data is None else arg.data #here is from util
    # read data
    data_train, data_val, data_test = read_data(arg.data)

    # create the model
    model = GTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=arg.context, num_tokens=NUM_TOKENS)
    if torch.cuda.is_available():
        model.cuda()

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())

    # training loop
    # - note: we don't loop over the data, instead we sample a batch of random subsequences each time.
    for i in tqdm.trange(arg.num_batches):
        # learning rate warmup
        # - we linearly increase the learning rate from 10e-10 to arg.lr over the first
        #   few thousand batches
        if arg.lr_warmup > 0 and i < arg.lr_warmup:
            lr = max(  (arg.lr / arg.lr_warmup) * i, 1e-10)
            opt.lr = lr

        opt.zero_grad()

        # sample a batch of random subsequences
        starts = torch.randint(size=(arg.batch_size, ), low=0, high=data_train.size(0) - arg.context - 1)

        seqs_source = [data_train[start  :start+arg.context  ] for start in starts]
        seqs_target = [data_train[start+1:start+arg.context+1] for start in starts]
        source = torch.cat([s[None, :] for s in seqs_source ], dim=0).to(torch.long)
        target = torch.cat([s[None, :] for s in seqs_target ], dim=0).to(torch.long)
        # - target is the same sequence as source, except one character ahead
        # source and target are two tensors, the dimension is: batch_size * seq_length (i.e., context)
        # that is to say, randomly extract batch_size sequence of characters with length context

        if torch.cuda.is_available():
            source, target = source.cuda(), target.cuda()
        
        output = model(source)

        loss = F.nll_loss(output.transpose(2, 1), target, reduction='mean')
        tbw.add_scalar('transformer/train-loss', float(loss.item()), i * arg.batch_size)

        loss.backward()

        # clip gradients
        # - If the total gradient vector has a length > 1, we clip it back down to 1.
        if arg.gradient_clipping > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

        opt.step()

        # we generate some random text to monitor progress every {arg.test_every} steps. 
        if i != 0 and (i % arg.test_every == 0 or i == arg.num_batches - 1): #test after every [test_every] iterations of batch training
            with torch.no_grad():
                # generate some random text
                GENSIZE = 600 #generate a string of length 600
                TEMP = 0.5 #temperature

                # randomly sample a sequence of length context from the data_test
                seedfr = random.randint(0, data_test.size(0) - arg.context)
                model_input = data_test[seedfr:seedfr + arg.context].to(torch.long)

                if torch.cuda.is_available():
                    model_input = model_input.cuda()

                print("\nThe sampled string is:")

                for c in model_input:
                    print(str(chr(c)), end='', flush=True) #print the sampled string

                for _ in range(GENSIZE):
                    output = model(model_input[None, :])
                    c = sample(output[0, -1, :], TEMP) #sample with temperature
                    print(str(chr(c)), end='', flush=True) #print the next generated character
                    model_input = torch.cat([model_input[1:], c[None]], dim=0) #get new model input to predict

                print("\n")

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-N", "--num-batches",
                        dest="num_batches",
                        help="Number of batches to train on. Each batch contains randomly sampled subsequences of the data.",
                        default=10000, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=4, type=int)

    parser.add_argument("-D", "--data", dest="data",
                        help="Data file.",
                        default=None)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.001, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='./runs')

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=32, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-C", "--context", dest="context",
                        help="Length of the sequences extracted from the corpus (and the context used during inference).",
                        default=256, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr of self-attention layers)",
                        default=4, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--test-every",
                        dest="test_every",
                        help="How many batches between tests.",
                        default=500, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=100, type=int)

    options = parser.parse_args()

    # print('OPTIONS ', options)

    go(options)