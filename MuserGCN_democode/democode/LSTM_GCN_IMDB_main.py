

import time
from util import *
import pickle
import yaml
import torch
from model.Build_model import build_LSTM_GCN_models, build_LSTM_GCN_models2
from Trainer.train import GCN_LSTM_Trainer, GCN_LSTM_Trainer2
from Trainer.Optimizer import build_GCN_LSTM_optimizers, build_GCN_LSTM_optimizers2
import random
from tqdm import tqdm


# Load data
with open('./dataset/imdb_graph_doc_train.pkl','rb') as f:
    imdb_data_train = pickle.load(f)
train_data = imdb_data_train['Doc']#[11500:13500]
train_A = imdb_data_train['A']#[11500:13500]
train_label = imdb_data_train['label']#[11500:13500]
train_num_node = imdb_data_train['num_node']#[11500:13500]
train_doc_length = imdb_data_train['doc_length']
train_BOW = imdb_data_train['train_BOW_data2']
train_mask = imdb_data_train['mask']
del imdb_data_train

with open('./dataset/imdb_graph_doc_test.pkl','rb') as f:
    imdb_data_test = pickle.load(f)
test_data = imdb_data_test['Doc']
test_A = imdb_data_test['A']
test_label = imdb_data_test['label']
test_num_node = imdb_data_test['num_node']
test_doc_length = imdb_data_test['doc_length']
test_BOW = imdb_data_test['test_BOW_data2']
test_mask = imdb_data_test['mask']
del imdb_data_test

train_num = len(train_data)
test_num = len(test_data)

# config
config_path = './configs/IMDB_LSTM_GCN.yaml'
with open(config_path, 'r') as f:
    config = yaml.load(f)

#
is_cuda = torch.cuda.is_available()

if is_cuda:
    device = config['gpu']['device']
else:
    device = "cpu"

Iterationall = config['training']['epoch'] * (train_num // config['training']['batch_size'])

# Create models
GCN_model, LSTM_model, PGBN_encoder = build_LSTM_GCN_models(config, Iterationall)
#GCN_model1, GCN_model2, LSTM_model, PGBN_encoder = build_LSTM_GCN_models2(config, Iterationall)

# trained_model_path = './our_trained_pgbn2/model400' + '.pth'
# PGBN_model = torch.load(trained_model_path)
# PGBN_encoder.load_state_dict(PGBN_model['PGBN_encoder'])


GCN_model = GCN_model.to(device)
LSTM_model = LSTM_model.to(device)
PGBN_encoder = PGBN_encoder.to(device)

GCN_optimizer, LSTM_optimizer, PGBN_optimizer = build_GCN_LSTM_optimizers(GCN_model, LSTM_model, PGBN_encoder, config)
#GCN1_optimizer, GCN2_optimizer, LSTM_optimizer, PGBN_optimizer = build_GCN_LSTM_optimizers2(GCN_model1, GCN_model2, LSTM_model, PGBN_encoder, config)

# Trainer
trainer = GCN_LSTM_Trainer(GCN_model, LSTM_model, PGBN_encoder, GCN_optimizer, LSTM_optimizer, PGBN_optimizer)
#trainer = GCN_LSTM_Trainer2(GCN_model1, GCN_model2, LSTM_model, PGBN_encoder, GCN1_optimizer, GCN2_optimizer, LSTM_optimizer, PGBN_optimizer)

Likelihood_all = []
Classification_all = []
Loss_all = []

## allocate space
A_minibatch = np.zeros([config['training']['batch_size'], config['data']['train_num_node'], config['data']['train_num_node']], dtype=np.float32)
test_A_minibatch = np.zeros([config['training']['eval_batch_size'], config['data']['test_num_node'], config['data']['test_num_node']], dtype=np.float32)


print('Start training...')

if config['model']['SentenceEmbed'] == 'LSTM':
    dimNodeEmbedding = config['model']['im_LSTMh']
elif config['model']['SentenceEmbed'] == 'BiLSTM':
    dimNodeEmbedding = config['model']['im_LSTMh']*2


for epoch in range(config['training']['epoch']):
    print('Start epoch %d...' % epoch)
    tstart = time.time()
    Likelihood_epoch = 0
    Classification_epoch = 0
    Loss_epoch = 0

    ## disorder the dataset
    LIST = list(zip(train_data, train_A, train_mask, train_BOW, train_label, train_doc_length))
    random.shuffle(LIST)
    train_data[:], train_A[:], train_mask[:], train_BOW[:], train_label[:], train_doc_length[:] = zip(*LIST)

    for iteration in range(train_num // config['training']['batch_size']):
        data_minibatch = train_data[iteration * config['training']['batch_size']: (iteration + 1) * config['training']['batch_size']]
        label_minibatch = train_label[iteration * config['training']['batch_size']: (iteration + 1) * config['training']['batch_size']]
        BOW_minibatch_list = train_BOW[iteration * config['training']['batch_size']: (iteration + 1) * config['training']['batch_size']]
        mask_minibatch = train_mask[iteration * config['training']['batch_size']: (iteration + 1) * config['training']['batch_size']]
        data_minibatch_length = train_doc_length[iteration * config['training']['batch_size']: (iteration + 1) * config['training']['batch_size']]
        A_minibatch_list = train_A[iteration * config['training']['batch_size']: (iteration + 1) * config['training']['batch_size']]

        label_minibatch_T = torch.from_numpy(label_minibatch).to(torch.int64).to(device)

        tmp = np.array(mask_minibatch)
        mask_minibatch_T = torch.from_numpy(np.squeeze(tmp)).to(torch.float32).to(device)

        it = -1
        for A in A_minibatch_list:
            it += 1
            A_minibatch[it, :, :] = A.toarray()
            A_minibatch[A_minibatch < 0.2] = 0

        A_minibatch_T = torch.from_numpy(A_minibatch).to(device)

        Classificatioin_Loss, Likelihood, Loss= trainer.LSTM_GCN_model_trainstep(data_minibatch, BOW_minibatch_list, A_minibatch_T, mask_minibatch_T, label_minibatch_T, data_minibatch_length, device, epoch, dimNodeEmbedding=dimNodeEmbedding)
        print('[epoch %0d, it %4d] classificatioin_Loss = %.4f, Likelihood = %.4f, Loss = %.4f'
              % (epoch, iteration, Classificatioin_Loss, Likelihood, Loss))
        Likelihood_epoch += Likelihood
        Classification_epoch += Classificatioin_Loss
        Loss_epoch += Loss


    tend = time.time()
    print('[epoch %0d time %4f] classificatioin_Loss = %.4f, Likelihood = %.4f, Loss = %.4f'
          % (epoch, tend - tstart, Classification_epoch, Likelihood_epoch, Loss_epoch))

    #Model = {'Phi': PGBN_encoder.Phi, 'PGBN_encoder': PGBN_encoder.state_dict(), 'LSTM': LSTM_model.state_dict(), 'GCN': GCN_model.state_dict()}
    #torch.save(Model, './our_trained_model/Model' + str(epoch + 1) + '.pth')

    Likelihood_all.append(Likelihood_epoch)
    Classification_all.append(Classification_epoch)
    Loss_all.append(Loss_epoch)

    #Evaluation on test dataset
    Num_accuracy = 0
    for iteration in tqdm(range(test_num // config['training']['eval_batch_size'])):
        data_minibatch = test_data[iteration * config['training']['eval_batch_size']: (iteration + 1) * config['training']['eval_batch_size']]
        label_minibatch = test_label[iteration * config['training']['eval_batch_size']: (iteration + 1) * config['training']['eval_batch_size']]
        BOW_minibatch_list = test_BOW[iteration * config['training']['eval_batch_size']: (iteration + 1) * config['training']['eval_batch_size']]
        mask_minibatch = test_mask[iteration * config['training']['eval_batch_size']: (iteration + 1) * config['training']['eval_batch_size']]
        data_minibatch_length = test_doc_length[iteration * config['training']['eval_batch_size']: (iteration + 1) * config['training']['eval_batch_size']]
        A_minibatch_list = test_A[iteration * config['training']['eval_batch_size']: (iteration + 1) * config['training']['eval_batch_size']]

        label_minibatch_T = torch.from_numpy(label_minibatch).to(torch.int64).to(device)

        tmp = np.array(mask_minibatch)
        mask_minibatch_T = torch.from_numpy(np.squeeze(tmp)).to(torch.float32).to(device)

        it = -1
        for A in A_minibatch_list:
            it += 1
            test_A_minibatch[it, :, :] = A.toarray()
            test_A_minibatch[test_A_minibatch < 0.2] = 0

        A_minibatch_T = torch.from_numpy(test_A_minibatch).to(device)


        num_accuracy = trainer.eval_step(data_minibatch, BOW_minibatch_list, A_minibatch_T, mask_minibatch_T, label_minibatch_T, data_minibatch_length, device, dimNodeEmbedding=dimNodeEmbedding)
        Num_accuracy += num_accuracy

    print('[epoch %0d ] test accuracy = %.4f' % (epoch, Num_accuracy/test_num))

