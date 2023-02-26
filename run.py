import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from model import Model
from utils import *
import scipy.io as sio

from sklearn.metrics import roc_auc_score, average_precision_score
import random
import os

import argparse
from tqdm import tqdm

from torch.nn.utils import clip_grad_norm_

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set argument
parser = argparse.ArgumentParser(description='AEHCL')
parser.add_argument('--dataset', type=str, default='Aminer')  #
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--drop_out', type=float, default=0.2)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--num_epoch', type=int, default=3)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')  #
parser.add_argument('--auc_test_rounds', type=int, default=3)#
parser.add_argument('--negsamp_ratio', type=int, default=1)

parser.add_argument('--in_dim', type=int, default=108)
parser.add_argument('--out_dim', type=int, default=64)
parser.add_argument('--num_relations', type=int, default=2)
parser.add_argument('--num_of_attention_heads', type=int, default=4)
parser.add_argument('--ab', type=float, default=1)
parser.add_argument('--alpha', type=float, default=0.8)
parser.add_argument('--beta', type=float, default=0.2)
parser.add_argument('--t', type=float, default=1)

args = parser.parse_args()

os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False

inits = init()

author_label, label_author = inits.getEntityLabel('./data_event/author_label')
conf_label, label_conf = inits.getEntityLabel('./data_event/conf_label')
paper_label, label_paper = inits.getEntityLabel('./data_event/paper_label')
entity_label = (author_label, paper_label, conf_label, label_author, label_paper, label_conf)
#print(len(label_author[1]))

paper_neg_set, conf_neg_set, author_neg_set = inits.getAllNeg('./data_event/neg_entity.npy')
all_neg_set = (paper_neg_set, conf_neg_set, author_neg_set)

event_pos_set, event_neg_set = inits.getAllPosNegEvent('./data_event/posneg_event.npy')
all_event_set = (event_pos_set, event_neg_set)


#已经构造完异常的数据
sample_list, length = inits.getSamples('./data_event/sampleinstance_hyper_injected')
max_len = np.max(np.sum(length, -1))+1
sample_len = np.sum(length, -1)+1
#print(sample_len)
# headList, midList, tailList, headSet, midSet, tailSet, \
#     headSetList, midSetList, tailSetList = inits.getTriples('./data_event/sampleinstance_mp1_injected')

#sample_data = (headList, midList, tailList, headSet, midSet, tailSet, headSetList, midSetList, tailSetList)



#tripleTotal, entityTotal, headTotal, midTotal, tailTotal = inits.getGlobalValues()

features = inits.getNodeFeatures('./data_event/outputfeatures_mp1')#(16万,dim)

#label
temp = sio.loadmat('./data_event/labels_injected_hyper.mat')
label = np.squeeze(temp['label'])#


cnt_wait = 0
best = 1e9
best_t = 0
batch_size = args.batch_size
#percent = args.percent
alpha = args.alpha
beta = args.beta

train_size = len(sample_list)
train_batch_num = train_size // batch_size + 1
test_size = train_size
test_batch_num = train_batch_num
neg_sum=10



# Train and testmodel
ap_list = []
auc_list = []
event, entity_type = inits.generate_samples(sample_list, length, max_len)
for i in range(10):#mean and std

    model = Model(args, max_len)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) #eps=1e-3
    #optimiser = torch.optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    #optimiser = torch.optim.SGD(model.parameters(), lr=args.lr)

    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,mode='min',factor=0.8)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()#

    with tqdm(total=args.num_epoch*train_batch_num) as pbar:
        pbar.set_description('Training')

        for epoch in range(args.num_epoch):

            loss_full_batch = torch.zeros((train_size, 1))
            if torch.cuda.is_available():
                loss_full_batch = loss_full_batch.cuda()

            model.train()

            all_idx = list(range(train_size))
            random.shuffle(all_idx)
            total_loss = 0.


            #为这一个epoch构造正负样本 event:(73956, 3), neg_event_entity:(73956, 3), pos_event,(73956, 3)

            neg_context, neg_entity, pos_event, neg_event = inits.generate_neg_samples(sample_list, entity_label, all_neg_set, all_event_set, neg_sum)


            for batch_idx in range(train_batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (train_batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)
                cur_event = event[idx]#(cur_batch_size,3)
                cur_neg_context = neg_context[idx]
                cur_neg_entity = neg_entity[idx]
                cur_entity_type = entity_type[idx]
                cur_sample_len = sample_len[idx]
                #cur_neg_event_entity = neg_event_entity[idx]#(cur_batch_size,3)
                cur_pos_event = pos_event[idx]#(cur_batch_size,3)
                cur_neg_event = neg_event[idx]  # (cur_batch_size,3)
                # if(batch_idx==0):
                #     cur_neg_event = random.sample(list(event), cur_batch_size)
                # else:
                #     cur_neg_event = pre_pos_event[:cur_batch_size]
                # pre_pos_event = cur_pos_event

                event_features = inits.get_features(cur_event, features, max_len, neg_sum)
                neg_context_features = inits.get_features(cur_neg_context, features, max_len, neg_sum)#(batch, max_len, 108)
                neg_entity_features = inits.get_features(cur_neg_entity, features, max_len, neg_sum)
                # neg_event_entity_features = inits.get_features(cur_neg_event_entity, features)
                pos_event_features = inits.get_features(cur_pos_event, features, max_len, neg_sum)
                neg_event_features = inits.get_features(cur_neg_event, features, max_len, neg_sum)

                event_features = torch.FloatTensor(event_features)  # (cur_batch, 3, feature dim)
                neg_context_features = torch.FloatTensor(neg_context_features)
                neg_entity_features = torch.FloatTensor(neg_entity_features)
                cur_entity_type = torch.LongTensor(cur_entity_type)
                cur_sample_len = torch.LongTensor(cur_sample_len)
                # neg_event_entity_features = torch.FloatTensor(neg_event_entity_features)  # (cur_batch, 3, feature dim)
                pos_event_features = torch.FloatTensor(pos_event_features)  # (cur_batch, 3, feature dim)
                neg_event_features = torch.FloatTensor(neg_event_features)  # (cur_batch, 3, feature dim)

                # lbl_intra是事件内对比损失的label：（6*batch，1），2*batch pos, 4*batch neg
                lbl_intra = torch.unsqueeze(
                    torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size))), 1)
                # lbl_inter是事件间对比损失的label：（2*batch，1），1*batch pos, 1*batch neg
                lbl_inter = torch.unsqueeze(
                    torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size))), 1)

                lbl_intra_t = torch.unsqueeze(
                    torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size))), 1)


                if torch.cuda.is_available():
                    lbl_intra = lbl_intra.cuda()
                    lbl_inter = lbl_inter.cuda()
                    lbl_intra_t = lbl_intra_t.cuda()
                    event_features = event_features.cuda()
                    neg_context_features = neg_context_features.cuda()
                    neg_entity_features = neg_entity_features.cuda()
                    cur_entity_type = cur_entity_type.cuda()
                    # neg_event_entity_features = neg_event_entity_features.cuda()
                    pos_event_features = pos_event_features.cuda()
                    neg_event_features = neg_event_features.cuda()

                if torch.cuda.is_available():
                    # binary cross-entropy(BCE)
                    BCE_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda())
                else:
                    BCE_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))

                intra_t_logits, intra_loss, inter_logits, _,_,_ = model(args=args, event=event_features, pos_event=pos_event_features,
                                                   neg_event=neg_event_features, neg_context=neg_context_features,
                                                   neg_entity=neg_entity_features, entity_type=cur_entity_type,
                                                   sample_len=cur_sample_len)

                #intra_loss_all = BCE_loss(intra_logits, lbl_intra)#(6*batch,1) 每一对pair的交叉熵损失ylogs
                #intra_loss_all = BCE_loss(intra_logits, lbl_intra)
                intra_loss_all = intra_loss
                intra_t_loss_all = BCE_loss(intra_t_logits, lbl_intra_t)
                inter_loss_all = BCE_loss(inter_logits, lbl_inter)#(2*batch,1)

                intra_loss = torch.mean(intra_loss_all)
                intra_t_loss = torch.mean(intra_t_loss_all)
                inter_loss = torch.mean(inter_loss_all)

                #print(intra_loss)
                loss = args.ab*intra_loss+alpha*intra_t_loss+beta*inter_loss#intra_loss + beta *  #+ alpha *inter_loss#

                loss.backward()

                # clip_grad_norm_(model.parameters(), max_norm=30, norm_type=2)


                optimiser.step()
                #scheduler.step(loss)

                loss = loss.detach().cpu().numpy()
                #loss_full_batch[idx] = loss_all[: cur_batch_size].detach()  # loss_full_batch:(16484,1)

                if not is_final_batch:
                    total_loss += loss
                mean_loss = total_loss/(batch_idx+1)

                pbar.set_postfix(loss=mean_loss)
                pbar.update(1)

            mean_loss = (total_loss * batch_size + loss * cur_batch_size) / train_size

            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'best_model.pkl')
            else:
                cnt_wait += 1


            # pbar.set_postfix(loss=mean_loss)
            # pbar.update(1)
            print("Epoch:", '%04d' % (epoch + 1), "avg_loss={:.9f}".format(mean_loss))

    # Test model
    print('Loading {}th epoch'.format(best_t))
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('best_model.pkl'), False)

    multi_round_ano_score = np.zeros((args.auc_test_rounds, test_size)) #记录每一次采样每一个event的得分
    multi_round_ano_score_intra = np.zeros((args.auc_test_rounds, test_size))
    multi_round_ano_score_intra_t = np.zeros((args.auc_test_rounds, test_size))

    with tqdm(total=args.auc_test_rounds) as pbar_test:
        pbar_test.set_description('Testing')
        #pre_pos_event=[]
        for round in range(args.auc_test_rounds):

            all_idx = list(range(test_size))
            #random.shuffle(all_idx)
            #neg_context, neg_entity, pos_event, neg_event = inits.generate_neg_samples(sample_list, entity_label, all_neg_set, all_event_set, neg_sum)

            model.eval()
            # 为这一个epoch构造正负样本 event:(73956, 3), neg_event_entity:(73956, 3), pos_event,(73956, 3)
            for batch_idx in range(test_batch_num):

                optimiser.zero_grad()
                is_final_batch = (batch_idx == (test_batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)
                cur_event = event[idx]  # (cur_batch_size,3)
                cur_entity_type = entity_type[idx]
                cur_sample_len = sample_len[idx]
                cur_pos_event = pos_event[idx]  # (cur_batch_size,3)
                # cur_neg_event = neg_event[idx]  # (cur_batch_size,3)
                # cur_neg_context = neg_context[idx]
                # cur_neg_entity = neg_entity[idx]  # (cur_batch_size,3)


                event_features = inits.get_features(cur_event, features, max_len, neg_sum)
                pos_event_features = inits.get_features(cur_pos_event, features, max_len, neg_sum)
                # neg_event_features = inits.get_features(cur_neg_event, features, max_len, neg_sum)
                # neg_context_features = inits.get_features(cur_neg_context, features, max_len, neg_sum)
                # neg_entity_features = inits.get_features(cur_neg_entity, features, max_len, neg_sum)

                event_features = torch.FloatTensor(event_features)  # (cur_batch, feature dim)
                cur_entity_type = torch.LongTensor(cur_entity_type)
                cur_sample_len = torch.LongTensor(cur_sample_len)
                pos_event_features = torch.FloatTensor(pos_event_features)  # (cur_batch, feature dim)
                # neg_event_features = torch.FloatTensor(neg_event_features)  # (cur_batch, feature dim)
                # neg_context_features = torch.FloatTensor(neg_context_features)
                # neg_entity_features = torch.FloatTensor(neg_entity_features)

                if torch.cuda.is_available():
                    # lbl_intra.cuda()
                    # lbl_inter.cuda()
                    event_features = event_features.cuda()
                    cur_entity_type = cur_entity_type.cuda()
                    pos_event_features = pos_event_features.cuda()
                    # neg_event_features = neg_event_features.cuda()
                    # neg_context_features = neg_context_features.cuda()
                    # neg_entity_features = neg_entity_features.cuda()

                with torch.no_grad():#
                    #
                    # _,_,_, inter_neg, intra_t_neg, intra_max_loss = model(args=args, event=event_features, pos_event=pos_event_features,
                    #                                neg_event=neg_event_features, neg_entity=neg_entity_features, neg_context=neg_context_features,
                    #                                                           entity_type=cur_entity_type, sample_len=cur_sample_len, is_test=False) #(batch)
                    intra_t_pos, intra_pos, inter_pos = model(args=args, event=event_features, pos_event=pos_event_features,
                                                                              entity_type=cur_entity_type, sample_len=cur_sample_len, is_test=True) #(batch)
                    intra_pos_logits = torch.sigmoid(torch.squeeze(intra_pos))#(cur_batch,)
                    inter_pos_logits = torch.sigmoid(torch.squeeze(inter_pos))
                    intra_pos_t_logits = torch.sigmoid(torch.squeeze(intra_t_pos))
                    # intra_t_neg_logits = torch.sigmoid(torch.squeeze(intra_t_neg))
                    # inter_neg_logits = torch.sigmoid(torch.squeeze(inter_neg))

                intra_ano_score = (-intra_pos_logits).cpu().numpy()  # eq.(12)
                inter_ano_score = (-inter_pos_logits).cpu().numpy()  # eq.(12)
                intra_ano_t_score = (-intra_pos_t_logits).cpu().numpy()  # eq.(12)
                #print(args.ab*intra_ano_score[0], alpha*intra_ano_t_score[0], beta*inter_ano_score[0])
                ano_score = args.ab*intra_ano_score+alpha*intra_ano_t_score+beta*inter_ano_score#intra_ano_score+beta*intra_ano_t_score#intra_ano_score+beta*

                multi_round_ano_score[round, idx] = ano_score  # 这一轮 对应event的得分(round,test_size)
                #multi_round_ano_score_intra[round, idx] = intra_ano_score  # 这一轮 对应event的得分(round,test_size)
                #multi_round_ano_score_intra_t[round, idx] = intra_ano_t_score  # 这一轮 对应event的得分(round,test_size)

            pbar_test.update(1)

    ano_score_final = np.mean(multi_round_ano_score, axis=0) #eq.(12) 求平均 #(test_size,)

    ano_label = label#(3697,)

    ap = average_precision_score(ano_label, ano_score_final)
    auc = roc_auc_score(ano_label, ano_score_final)


    print('AP:{:.4f}'.format(ap))
    print('AUC:{:.4f}'.format(auc))


    ap_list.append(ap)
    auc_list.append(auc)
ap_avg = np.mean(ap_list)
ap_std = np.std(ap_list)
auc_avg = np.mean(auc_list)
auc_std = np.std(auc_list)
import time  # 引入time模块
ticks = time.time()
with open ('result.log','a+') as f:
    f.write('time:'+str(ticks)+'  '+'ours: '+'ap:'+str(ap_avg)+"+-"+str(ap_std)+" "+'auc:'+str(auc_avg)+"+-"+str(auc_std)+'\n')

    #import scipy.io as si0
    #sio.savemat('./data_event/logits', {'intra': multi_round_ano_score_intra, 'intra_t': multi_round_ano_score_intra_t})










