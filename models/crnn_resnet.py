import torch.nn as nn
import models.resnet as resnet

class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    #                   32    1   37     256
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.cnn = resnet.resnet34()
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        #print('---forward propagation---')
        conv = self.cnn(input)
        # print(conv.size()) #batch_size*512*1*with
        b, c, h, w = conv.size()
        assert w == 1, "the height of conv must be 1"
        conv = conv.squeeze(3) # b *512 * width
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        #print(conv.size()) # width batch_size channel
        # rnn features
        output = self.rnn(conv)
        #print(output.size(0))
        # print(output.size())# width*batch_size*nclass
        return output
