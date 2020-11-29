import torch.nn as nn
from torch.nn import functional as F
import torch
from torch_geometric.nn import DenseGCNConv
import numpy as np
from pgbn_tool import PGBN_sampler


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()

        self.gc1 = DenseGCNConv(nfeat, nhid)
        self.gc2 = DenseGCNConv(nhid, nhid)
        self.gc3 = DenseGCNConv(nfeat, nhid)
        self.gc4 = DenseGCNConv(nhid, nhid)
        self.dropout = dropout
        self.number_attention = 1
        self.classifier = nn.Linear(nhid*self.number_attention, 2)
       # self.classifier2 = nn.Linear(nhid*self.number_attention, 2)

        self.node_classifier = nn.Linear(nhid, 2)
        self.attention = nn.Linear(nhid, self.number_attention)

#     def forward(self, x, adj, mask, x_new, adj_new, mask_new):
#         x1 = F.relu(self.gc1(x, adj, mask, add_loop=False))
#         x2 = F.dropout(x1, self.dropout, training=self.training)
#         x2 = F.relu(self.gc2(x2, adj, mask, add_loop=False))
#         x2 = F.dropout(x2, self.dropout, training=self.training)
#
#         # self-attention
#         attn = torch.softmax(self.attention1(x2) * mask.unsqueeze(2).repeat(1,1,8), 1)
#         graph_feature = torch.bmm(x2.permute(0,2,1), attn).view(x.shape[0], -1)
#
#         # global pooling to get graph feature
#
#         #graph_feature1 = torch.max(x1, dim=1)[0]
#         #graph_feature2 = torch.max(x2, dim=1)[0]
#         #graph_feature = torch.cat([torch.max(x2, dim=1)[0], torch.sum(x2,dim=1) / torch.sum(mask,dim=1,keepdim=True)], dim=1)
#         #graph_feature = torch.cat([graph_feature1, graph_feature2], dim=1)
#         output1 = self.classifier1(graph_feature)
#         node_output = self.node_classifier(x2)
# ################################################################################################
#
#         x1 = F.relu(self.gc3(x_new, adj_new, mask_new, add_loop=True))
#         x2 = F.dropout(x1, self.dropout, training=self.training)
#         x2 = F.relu(self.gc4(x2, adj_new, mask_new, add_loop=True))
#         x2 = F.dropout(x2, self.dropout, training=self.training)
#
#         # self-attention
#         attn = torch.softmax(self.attention2(x2) * mask_new.unsqueeze(2).repeat(1, 1, 8), 1)
#         graph_feature = torch.bmm(x2.permute(0, 2, 1), attn).view(x.shape[0], -1)
#
#         #graph_feature1 = torch.max(x1, dim=1)[0]
#         #graph_feature2 = torch.max(x2, dim=1)[0]
#         #graph_feature = torch.cat([graph_feature1, graph_feature2], dim=1)
#         #graph_feature = torch.cat([torch.max(x2, dim=1)[0], torch.sum(x2,dim=1) / torch.sum(mask_new,dim=1,keepdim=True)], dim=1)
#
#         output2 = self.classifier2(graph_feature)
#         return output1, output2, node_output

    def forward(self, x, adj, mask, x_new, adj_new, mask_new):
        x1 = F.relu(self.gc1(x, adj, mask, add_loop=False))
        x2 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x2, adj, mask, add_loop=False))
        x2_1 = F.dropout(x2, self.dropout, training=self.training)


        x1 = F.relu(self.gc3(x_new, adj_new, mask_new, add_loop=True))
        x2 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc4(x2, adj_new, mask_new, add_loop=True))
        x2_2 = F.dropout(x2, self.dropout, training=self.training)

        x2_joint = torch.cat([x2_1, x2_2], dim=1)
        mask_joint = torch.cat([mask, mask_new], dim=1)

        # self-attention
        attn = torch.softmax(self.attention(x2_joint) * mask_joint.unsqueeze(2).repeat(1, 1, self.number_attention), 1)
        graph_feature = torch.bmm(x2_joint.permute(0, 2, 1), attn).view(x.shape[0], -1)

        #graph_feature1 = torch.max(x1, dim=1)[0]
        #graph_feature2 = torch.max(x2, dim=1)[0]
        #graph_feature = torch.cat([graph_feature1, graph_feature2], dim=1)
        #graph_feature = torch.cat([torch.max(x2, dim=1)[0], torch.sum(x2,dim=1) / torch.sum(mask_new,dim=1,keepdim=True)], dim=1)
        node_output = self.node_classifier(x2_joint)
        output1 = self.classifier(graph_feature)
        output2 = 0
        return output1, output2, node_output

class GCN1(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN1, self).__init__()

        self.gc1 = DenseGCNConv(nfeat, nhid)
        self.gc2 = DenseGCNConv(nhid, nhid)
        self.dropout = dropout
        self.classifier = nn.Linear(nhid*2, 2)
        self.attention = nn.Linear(nhid, 8)
        self.node_classifier = nn.Linear(nhid, 2)

    def forward(self, x, adj, mask):
        x1 = F.relu(self.gc1(x, adj, mask, add_loop=False))
        x2 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x2, adj, mask, add_loop=False))
        x2 = F.dropout(x2, self.dropout, training=self.training)

        # attn_tmp = torch.exp(self.attention(x)) * mask.unsqueeze(2).repeat(1,1,8)
        # attn = torch.zeros([x.shape[0],attn_tmp.shape[1],attn_tmp.shape[2]]).to('cuda')
        # for i in range(8):
        #     aa = attn_tmp[:,:,i]
        #     attn[:, :, i] = (aa.t()/aa.sum(1)).t()
        # graph_feature = torch.bmm(x.permute(0,2,1), attn).view(x.shape[0], -1)

        graph_feature1 = torch.max(x1, dim=1)[0]
        graph_feature2 = torch.max(x2, dim=1)[0]
        graph_feature = torch.cat([graph_feature1, graph_feature2], dim=1)
        output = self.classifier(graph_feature)
        node_output1 = self.node_classifier(x1)
        node_output2 = self.node_classifier(x2)

        return output, node_output1, node_output2

class GCN2(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN2, self).__init__()

        self.gc1 = DenseGCNConv(nfeat, nhid)
        self.gc2 = DenseGCNConv(nhid, nhid)
        self.dropout = dropout
        self.classifier = nn.Linear(nhid*2, 2)


    def forward(self, x, adj, mask):
        x1 = F.relu(self.gc1(x, adj, mask, add_loop=False))
        x2 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x2, adj, mask, add_loop=False))
        x2 = F.dropout(x2, self.dropout, training=self.training)

        graph_feature1 = torch.max(x1, dim=1)[0]
        graph_feature2 = torch.max(x2, dim=1)[0]
        graph_feature = torch.cat([graph_feature1, graph_feature2], dim=1)
        output = self.classifier(graph_feature)

        return output

class LSTM(nn.Module):
    def __init__(self, vocab_size, dim_wordE, dim_LSTMh):
        super(LSTM, self).__init__()

        self.embed = nn.Embedding(vocab_size, dim_wordE, padding_idx=vocab_size-1)

        self.lstm = nn.LSTM(input_size=dim_wordE,
                            hidden_size=dim_LSTMh,
                            num_layers=1,
                            batch_first=True,
                            dropout=0)



    def forward(self, x, length, device):

        doc = torch.from_numpy(x).to(device).long()
        word_embed = self.embed(doc)

        word_embed = F.dropout(word_embed, 0.3, training=self.training)

        hidden_units, (_, _) = self.lstm(word_embed)

        feature = torch.zeros(1, len(length), hidden_units.shape[2]).to(device)
        for i in range(len(length)):
            ll = length[i]
            #feature[0,i,:] = hidden_units[i, ll-1, :]
            feature[0, i, :] = torch.max(hidden_units[i, :ll, :], dim=0)[0]
        return feature

class OnlyLSTM(nn.Module):
    def __init__(self, vocab_size, dim_wordE, dim_LSTMh):
        super(OnlyLSTM, self).__init__()

        self.embed = nn.Embedding(vocab_size, dim_wordE, padding_idx=vocab_size-1)

        self.lstm = nn.LSTM(input_size=dim_wordE,
                            hidden_size=dim_LSTMh,
                            num_layers=1,
                            batch_first=True,
                            dropout=0)
        self.classifier = nn.Linear(dim_LSTMh, 2)



    def forward(self, x, length, device):
        word_embed = self.embed(x)

        hidden_units, (_, _) = self.lstm(word_embed)

        feature = torch.zeros(len(length), hidden_units.shape[2]).to(device)
        for i in range(len(length)):
            ll = np.int(length[i][0])
            #feature[0,i,:] = hidden_units[i, ll-1, :]
            feature[i, :] = torch.max(hidden_units[i, :ll, :], dim=0)[0]

        output = self.classifier(feature)
        return output

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, dim_wordE, dim_LSTMh):
        super(BiLSTM, self).__init__()

        self.embed = nn.Embedding(vocab_size, dim_wordE, padding_idx=vocab_size - 1)

        self.lstm = nn.LSTM(input_size=dim_wordE, hidden_size= dim_LSTMh, num_layers= 2, batch_first=True, bidirectional=True)

    def forward(self, x, sentlen, device):
        doc = torch.from_numpy(x).to(device).long()
        sent_embed = self.embed(doc)

        sent_embed = F.dropout(sent_embed, 0.3, training=self.training)

        sentlen = torch.tensor(sentlen).to(device)

        idx_sort = torch.argsort(sentlen, descending = True)
        #print(idx_sort)
        sent_sorted = sent_embed.index_select(dim = 0, index = idx_sort)
        #print(sent_sorted)
        sentlen_sorted = sentlen.index_select(dim = 0, index = idx_sort)
        #print(sentlen_sorted)
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent_sorted, sentlen_sorted, batch_first = True)
        sent_output =  self.lstm(sent_packed)[0]
        sent_unpacked = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first = True)[0]
        idx_reorder = torch.argsort(idx_sort, descending = False)
        sent_reorder = sent_unpacked.index_select( dim = 0, index =idx_reorder)
        #sentlen_reorder = sentlen_sorted.index_select( dim =0, index = idx_reorder)

        feature = torch.zeros(1, len(sentlen), sent_reorder.shape[2]).to(device)
        for i in range(len(sentlen)):
            ll = sentlen[i]
            feature[0, i, :] = torch.max(sent_reorder[i, :ll, :], dim=0)[0]

        return feature

class OnlyBiLSTM(nn.Module):
    def __init__(self, vocab_size, dim_wordE, dim_LSTMh):
        super(OnlyBiLSTM, self).__init__()

        self.embed = nn.Embedding(vocab_size, dim_wordE, padding_idx=vocab_size - 1)

        self.lstm = nn.LSTM(input_size=dim_wordE, hidden_size= dim_LSTMh, num_layers= 2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(dim_LSTMh*2, 2)

    def forward(self, x, sentlen, device):
        sent_embed = self.embed(x)
        sent_embed = F.dropout(sent_embed, 0.3, training=self.training)

        sentlen = torch.tensor(sentlen.squeeze()).to(torch.int32).to(device)

        idx_sort = torch.argsort(sentlen, descending = True)

        sent_sorted = sent_embed.index_select(dim = 0, index = idx_sort)

        sentlen_sorted = sentlen.index_select(dim = 0, index = idx_sort)

        sent_packed = nn.utils.rnn.pack_padded_sequence(sent_sorted, sentlen_sorted, batch_first = True)
        sent_output = self.lstm(sent_packed)[0]
        sent_unpacked = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first = True)[0]
        idx_reorder = torch.argsort(idx_sort, descending = False)
        sent_reorder = sent_unpacked.index_select( dim = 0, index =idx_reorder)
        #sentlen_reorder = sentlen_sorted.index_select( dim =0, index = idx_reorder)

        feature = torch.zeros(len(sentlen), sent_reorder.shape[2]).to(device)
        for i in range(len(sentlen)):
            ll = sentlen[i]
            feature[i, :] = torch.max(sent_reorder[i, :ll, :], dim=0)[0]

        output = self.classifier(feature)
        return output

class PGBN(nn.Module):
    def __init__(self, voc_size, hDim, numTopic, device, Total_sentence, Iterationall):
        super(PGBN, self).__init__()
        self.voc_size = voc_size
        self.hDim = hDim
        self.numTopic = numTopic
        self.device = device

        self.f1 = nn.Linear(voc_size, hDim)
        self.shape = nn.Linear(hDim, 1)
        self.scale = nn.Linear(hDim, numTopic)

        trained_model_path = './our_trained_pgbn2/model400' + '.pth'

        PGBN_model = torch.load(trained_model_path)
        Phi = PGBN_model['Phi'].to('cpu').numpy()

        # realmin = 2.2e-308
        # Phi = 0.2 + 0.8 * np.random.rand(voc_size, self.numTopic)
        # Phi = Phi / np.maximum(realmin, Phi.sum(0))
        self.Phi = torch.from_numpy(Phi).float().to(device)

        self.train_PGBN_step = -1
        self.Total_sentence = Total_sentence

        self.pgbn_eta = 0.01

        Setting = {}
        Setting['Iterall'] = Iterationall
        Setting['tao0FR'] = 0
        Setting['kappa0FR'] = 0.9
        Setting['tao0'] = 20
        Setting['kappa0'] = 0.7
        Setting['epsi0'] = 1

        self.ForgetRate = np.power((Setting['tao0FR'] + np.linspace(1, Setting['Iterall'], Setting['Iterall'])),
                                   -Setting['kappa0FR'])
        epsit = np.power((Setting['tao0'] + np.linspace(1, Setting['Iterall'], Setting['Iterall'])), -Setting['kappa0'])
        self.epsit = Setting['epsi0'] * epsit / epsit[0]



    def forward(self, x):
        h = F.softplus(self.f1(torch.log(1+x)))
        Theta_shape = torch.exp(self.shape(h))
        #Theta_shape = torch.clamp(Theta_shape, min=1e-3, max=1e3)
        Theta_scale = torch.exp(self.scale(h))

        Theta_shape.repeat([1, self.numTopic])

        Theta_e = torch.Tensor(x.shape[0], self.numTopic).uniform_(0,1).to(self.device)
        Theta = (Theta_scale * ((-torch.log(1 - Theta_e)) ** (1 / Theta_shape)))


        return Theta

    def updatePhi(self, MBratio, MBObserved, X, Theta):
        Theta_np = Theta.t().to('cpu').detach().numpy()

        Xt = X.t().to('cpu').numpy()
        phi = self.Phi.to('cpu').numpy()
        Xt_to_t1, WSZS = PGBN_sampler.Multrnd_Matrix(Xt.astype('double'), phi.astype('double'), Theta_np.astype('double'))

        EWSZS = WSZS
        EWSZS = MBratio * EWSZS

        if (MBObserved == 0):
            self.NDot = EWSZS.sum(0)
        else:
            self.NDot = (1 - self.ForgetRate[MBObserved]) * self.NDot + self.ForgetRate[MBObserved] * EWSZS.sum(0)
        tmp = EWSZS + self.pgbn_eta
        tmp = (1 / self.NDot) * (tmp - tmp.sum(0) * phi)
        tmp1 = (2 / self.NDot) * phi

        tmp = phi + self.epsit[MBObserved] * tmp + np.sqrt(self.epsit[MBObserved] * tmp1) * np.random.randn(
            phi.shape[0], phi.shape[1])
        phi = PGBN_sampler.ProjSimplexSpecial(tmp, phi, 0)
        self.Phi = torch.from_numpy(phi).float().to(self.device)



def build_LSTM_GCN_models(config, Iterationall):
    if config['model']['SentenceEmbed'] == 'LSTM':
        LSTM_model = LSTM(vocab_size = config['data']['vocab_size'],
                    dim_wordE = config['data']['dim_wordE'],
                    dim_LSTMh = config['model']['im_LSTMh'])

        GCN_model = GCN(nfeat=config['model']['im_LSTMh'],
                        nhid=config['model']['nhid'],
                        dropout=config['model']['dropout'])

    elif config['model']['SentenceEmbed'] == 'BiLSTM':
        LSTM_model = BiLSTM(vocab_size=config['data']['vocab_size'],
                            dim_wordE=config['data']['dim_wordE'],
                            dim_LSTMh=config['model']['im_LSTMh'])

        GCN_model = GCN(nfeat=config['model']['im_LSTMh']*2,
                        nhid=config['model']['nhid'],
                        dropout=config['model']['dropout'])

    PGBN_model = PGBN(voc_size = 29660,
                      hDim = config['model']['PGBN_h1'],
                      numTopic = config['model']['PGBN_numTopic1'],
                      device = config['gpu']['device'],
                      Total_sentence = config['data']['Total_num_sentence'],
                      Iterationall = Iterationall)







    return GCN_model, LSTM_model, PGBN_model

def build_LSTM_GCN_models2(config, Iterationall):
    if config['model']['SentenceEmbed'] == 'LSTM':
        LSTM_model = LSTM(vocab_size = config['data']['vocab_size'],
                    dim_wordE = config['data']['dim_wordE'],
                    dim_LSTMh = config['model']['im_LSTMh'])

        GCN_model1 = GCN1(nfeat=config['model']['im_LSTMh'],
                        nhid=config['model']['nhid'],
                        dropout=config['model']['dropout'])

        GCN_model2 = GCN2(nfeat=config['model']['nhid'],
                         nhid=config['model']['nhid'],
                         dropout=config['model']['dropout'])

    PGBN_model = PGBN(voc_size = 29660,
                      hDim = config['model']['PGBN_h1'],
                      numTopic = config['model']['PGBN_numTopic1'],
                      device = config['gpu']['device'],
                      Total_sentence = config['data']['Total_num_sentence'],
                      Iterationall = Iterationall)







    return GCN_model1, GCN_model2, LSTM_model, PGBN_model

def build_models(config):

    model = GCN(nfeat = config['data']['Dim_feat'],
                nhid = config['model']['nhid'],
                dropout = config['model']['dropout'])

    return model


def build_LSTM_models(config):
    if config['model']['SentenceEmbed'] == 'LSTM':
        LSTM_model = OnlyLSTM(vocab_size = config['data']['vocab_size'],
                    dim_wordE = config['data']['dim_wordE'],
                    dim_LSTMh = config['model']['im_LSTMh'])

    elif config['model']['SentenceEmbed'] == 'BiLSTM':
        LSTM_model = OnlyBiLSTM(vocab_size=config['data']['vocab_size'],
                            dim_wordE=config['data']['dim_wordE'],
                            dim_LSTMh=config['model']['im_LSTMh'])

    return LSTM_model

def build_LSTM_GCN_models_tmp(config):
    if config['model']['SentenceEmbed'] == 'LSTM':
        LSTM_model = LSTM(vocab_size = config['data']['vocab_size'],
                    dim_wordE = config['data']['dim_wordE'],
                    dim_LSTMh = config['model']['im_LSTMh'])

        GCN_model = GCN1(nfeat=config['model']['im_LSTMh'],
                        nhid=config['model']['nhid'],
                        dropout=config['model']['dropout'])

    elif config['model']['SentenceEmbed'] == 'BiLSTM':
        LSTM_model = BiLSTM(vocab_size=config['data']['vocab_size'],
                            dim_wordE=config['data']['dim_wordE'],
                            dim_LSTMh=config['model']['im_LSTMh'])

        GCN_model = GCN1(nfeat=config['model']['im_LSTMh']*2,
                        nhid=config['model']['nhid'],
                        dropout=config['model']['dropout'])

    return GCN_model, LSTM_model


