import torch
import torch.nn as nn
import math
import numpy as np
# from torch.autograd import Variable
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.ap_bilinear = nn.Bilinear(args.out_dim, args.out_dim, 1)#author_paper
        # self.cp_bilinear = nn.Bilinear(args.out_dim, args.out_dim, 1)#conf_paper

        self.ap_bilinear = nn.Parameter(torch.empty(size=(args.out_dim, args.out_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.ap_bilinear.data, gain=1.414)
        self.cp_bilinear = nn.Parameter(torch.empty(size=(args.out_dim, args.out_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.cp_bilinear.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        self.att_drop = nn.Dropout(0.5)




    def forward(self, event):
        #event: (batch, max_len, dim)
        #target: (batch, 1, dim)
        target = torch.unsqueeze(event[:, 0, :], 1)
        context = event[:, 1:, :]

        conf = torch.unsqueeze(context[:, 0, :], 1)
        author = context[:, 1:, :]
        #drop_out
        # cp_bilinear = self.att_drop(self.cp_bilinear)
        # ap_bilinear = self.att_drop(self.ap_bilinear)

        conf_ = conf.matmul(self.cp_bilinear)#batch ,1, dim
        conf_coef = conf_.matmul(torch.transpose(target, 1, 2))#batch, 1, 1
        author_ = author.matmul(self.ap_bilinear) #batch, max_len-2, dim
        author_coef = author_.matmul(torch.transpose(target, 1, 2))#batch, max_len-2, 1
        coef = torch.cat((conf_coef, author_coef), 1) #batch, max_len-1, 1
        coef = torch.squeeze(coef) #batch ,max_len-1
        att = self.leakyrelu(coef)
        att = self.softmax(att) #batch, max_len -1
        att = torch.unsqueeze(att, -1)
        context_aggre = (att*context).sum(dim=1) #batch, dim
        cur_event = torch.cat((context_aggre, torch.squeeze(target)),1) #batch ,dim*2

        return cur_event






class SelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_attention_heads = args.num_of_attention_heads
        self.attention_head_size = int(args.out_dim / args.num_of_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.out_dim, self.all_head_size)
        self.key = nn.Linear(args.out_dim, self.all_head_size)
        self.value = nn.Linear(args.out_dim, self.all_head_size)

        self.dense = nn.Linear(args.out_dim, args.out_dim)

        #self.type_embedding = nn.Embedding.from_pretrained(torch.FloatTensor([[0, 0], [0, 1], [1, 0]]), freeze=True)
        self.type_embedding = nn.Embedding(3, args.out_dim, padding_idx=0)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, entity_type):
        #type embedding

        # Get embeddings for index 1
        # entity_type = entity_type.long()
        type_embedding = self.type_embedding(entity_type) #batch, max_len, 2
        hidden_states = hidden_states+type_embedding
        #hidden_states = torch.cat((hidden_states, type_embedding), 2)  #batch, max_len, out_dim+2

        mixed_query_layer = self.query(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(hidden_states)  # [Batch_size x Seq_length x Hidden_size]

        query_layer = self.transpose_for_scores(
            mixed_query_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        value_layer = self.transpose_for_scores(
            mixed_value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
                                                                         -2))  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        context_layer = torch.matmul(attention_probs,
                                     value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        context_layer = context_layer.permute(0, 2, 1,
                                              3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (
        self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]

        output = self.dense(context_layer) #

        return output


class Model(nn.Module):
    def __init__(self, args, max_len):
        super(Model, self).__init__()
        #两种entity的转换

        self.fc_paper = nn.Sequential(
            nn.Linear(args.in_dim, args.out_dim, bias=True),
            nn.ELU(),
            nn.Linear(args.out_dim, args.out_dim, bias=True),
            #nn.LayerNorm([1, args.out_dim])
            #nn.Dropout(p=args.drop_out)

        )


        self.fc_author = nn.Sequential(
            nn.Linear(args.in_dim, args.out_dim, bias=True),
            nn.ELU(),
            nn.Linear(args.out_dim, args.out_dim, bias=True),

            #nn.Dropout(p=args.drop_out)
        )


        self.fc_conf = nn.Sequential(
            nn.Linear(args.in_dim, args.out_dim, bias=True),
            nn.ELU(),
            nn.Linear(args.out_dim, args.out_dim, bias=True),
            #nn.LayerNorm([1, args.out_dim])
            #nn.Dropout(p=args.drop_out)
        )

        for model in self.fc_paper:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

        for model in self.fc_conf:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

        for model in self.fc_author:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

        self.fc_paper_inter = nn.Sequential(
            nn.Linear(args.in_dim, args.out_dim, bias=True),
            nn.ELU(),
            nn.Linear(args.out_dim, args.out_dim, bias=True),
            #nn.LayerNorm([1, args.out_dim])
            #nn.Dropout(p=args.drop_out)
        )

        self.fc_author_inter = nn.Sequential(
            nn.Linear(args.in_dim, args.out_dim, bias=True),
            nn.ELU(),
            nn.Linear(args.out_dim, args.out_dim, bias=True),
            #nn.LayerNorm([1, args.out_dim])
            #nn.Dropout(p=args.drop_out)
        )

        self.fc_conf_inter = nn.Sequential(
            nn.Linear(args.in_dim, args.out_dim, bias=True),
            nn.ELU(),
            nn.Linear(args.out_dim, args.out_dim, bias=True),
            #nn.LayerNorm([1, args.out_dim])
            #nn.Dropout(p=args.drop_out)
        )

        for model in self.fc_paper_inter:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

        for model in self.fc_conf_inter:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

        for model in self.fc_author_inter:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)




        self.batch_norm1d = nn.BatchNorm1d(args.out_dim)
        self.batch_norm2d = nn.BatchNorm2d(args.out_dim)


        self.layer_norm1 = nn.LayerNorm([max_len, max_len])
        self.layer_norm2 = nn.LayerNorm([max_len, 10])


        self.fc3 = nn.Linear(args.out_dim*3, args.out_dim, bias=False)

        #不同关系下两个entity的bilinear
        self.bl = []
        for i in range(args.num_relations):
            self.bl.append(nn.Bilinear(args.out_dim, args.out_dim, 1))  # eq.(8): bilinear scoring function
            self.bl[i] = self.bl[i].cuda()

        #正负event的bilinear
        self.ebl = nn.Bilinear(args.out_dim*2, args.out_dim*2, 1)
        self.ebl_t = nn.Bilinear(args.out_dim, args.out_dim, 1)

        #self attention
        self.selfatt = SelfAttention(args)
        #maxpool
        self.maxpool = nn.MaxPool2d(kernel_size=(max_len-1, 1))

        #att
        self.att = Attention(args)


    #event: {batch, 3, dim} #neg_entity:(batch, max_len, max_neg_len, dim):每个entity找一个替换的entity
    # pos_event:{batch, 3, dim}  neg_event:{batch, 3, dim}
    def forward(self, args, event, pos_event, neg_event=None, neg_context=None, neg_entity=None, entity_type=None, sample_len=None, is_test=False):
        batch_size = len(event)
        max_len = event.shape[1]

        #transformation
        paper = event[:, 0, :] #batch,1,dim
        conf = event[:, 1, :]
        author = event[:, 2:, :]

        paper = self.fc_paper(paper)
        paper = torch.unsqueeze(paper, 1)#batch, 1, outdim

        conf = self.fc_conf(conf)
        conf = torch.unsqueeze(conf, 1)#batch, 1, outdim

        author = self.fc_author(author)#batch, len, outdim
        event = torch.cat((paper, conf, author), 1)


        pos_event_paper = pos_event[:, 0, :]
        pos_event_conf = pos_event[:, 1, :]
        pos_event_author = pos_event[:, 2:, :]

        pos_event_paper = self.fc_paper(pos_event_paper)
        pos_event_paper = torch.unsqueeze(pos_event_paper, 1)  # batch, 1, outdim

        pos_event_conf = self.fc_conf(pos_event_conf)
        pos_event_conf = torch.unsqueeze(pos_event_conf, 1)  # batch, 1, outdim

        pos_event_author = self.fc_author(pos_event_author)  # batch, len, outdim

        #inter_pos
        pos_event = torch.cat((pos_event_paper, pos_event_conf, pos_event_author), 1)
        pos_event = self.att(pos_event)# batch ,dim*2
        event = self.att(event)# batch ,dim*2
        pos_inter = self.ebl(pos_event, event)#batch ,1


        #intra_t_pos
        pos_context = torch.cat((conf, author), 1)
        pos_context = self.selfatt(pos_context, entity_type)  # batch, max_len-1, outdim
        pos_context = self.maxpool(pos_context)  # batch, 1, outdim
        pos_t = self.ebl_t(pos_context, paper)
        pos_t = torch.squeeze(pos_t, -1)

        # intra_pos
        pos_entity = torch.cat((paper, conf, author), 1)  # batch, max_len, dim
        pos_norm = torch.norm(pos_entity, dim=-1, keepdim=True)  # batch, max_len, 1
        pos_entity = pos_entity / pos_norm  # batch, max_len, dim
        # print(pos_entity[0][0])
        pos_matrix = torch.matmul(pos_entity, torch.transpose(pos_entity, 1, 2))  # (batch, max_len, max_len)
        mask = torch.ones_like(pos_matrix) - torch.eye(pos_matrix.shape[1]).cuda()
        pos_matrix = torch.mul(pos_matrix, mask)  # remove diag # (batch, max_len, max_len)
        #print(pos_matrix[2,:4,:4])



        if(is_test == True):
            #inter_test
            inter_pos = pos_inter

            # intra_t_test
            intra_t_pos = pos_t

            # intra_test
            '''1. min'''
            intra_pos = []
            mask_ = 1000000.0 * torch.eye(pos_matrix.shape[1]).cuda()
            pos_ = pos_matrix + mask_
            for i, sample in enumerate(pos_matrix):
                pos_i = pos_[i, :sample_len[i], :sample_len[i]]
                pos_i = torch.flatten(pos_i)
                min1 = torch.min(pos_i)
                intra_pos.append(min1)
            intra_pos = torch.unsqueeze(torch.FloatTensor(intra_pos), -1).cuda()
            '''2. avg'''
            # intra_pos = []
            # mask_ = torch.ones(pos_matrix.shape[1]).cuda() - torch.eye(pos_matrix.shape[1]).cuda()
            # pos_ = pos_matrix * mask_
            # for i, sample in enumerate(pos_matrix):
            #     pos_i = pos_[i, :sample_len[i], :sample_len[i]]
            #     pos_i = torch.flatten(pos_i)
            #     mean1 = torch.sum(pos_i)/(pos_i.shape[0]-sample_len[i])
            #     intra_pos.append(mean1)
            # intra_pos = torch.unsqueeze(torch.FloatTensor(intra_pos), -1).cuda()
            '''3. std'''
            # intra_pos = []
            # mask_ = torch.ones(pos_matrix.shape[1]).cuda() - torch.eye(pos_matrix.shape[1]).cuda()
            # pos_ = pos_matrix * mask_ #对角设为0
            # for i, sample in enumerate(pos_matrix):
            #     pos_i = pos_[i, :sample_len[i], :sample_len[i]]
            #     pos_i = torch.flatten(pos_i)
            #     mean1 = torch.mean(pos_i)
            #     std1 = torch.sum(torch.triu(torch.sqrt((pos_ - mean1)**2), diagonal=1))/(sample_len[i]*(sample_len[i]-1)/2.0)
            #     intra_pos.append(-std1)#方差越大越异常
            # intra_pos = torch.unsqueeze(torch.FloatTensor(intra_pos), -1).cuda()

            '''4. loss'''
            #在后面定义


            return intra_t_pos, intra_pos, inter_pos



        elif(is_test == False): #train
            #inter_loss
            neg_event_paper = neg_event[:, 0, :]
            neg_event_conf = neg_event[:, 1, :]
            neg_event_author = neg_event[:, 2:, :]

            neg_event_paper = torch.unsqueeze(self.fc_paper(neg_event_paper), 1)  # batch, 1, outdim
            neg_event_conf = torch.unsqueeze(self.fc_conf(neg_event_conf), 1)  # batch, 1, outdim
            neg_event_author = self.fc_author(neg_event_author)  # batch, len, outdim

            neg_event = torch.cat((neg_event_paper, neg_event_conf, neg_event_author), 1)
            neg_event = self.att(neg_event)
            neg_inter = self.ebl(neg_event, event)
            '''inter_neg test'''
            inter_neg = neg_inter
            inter_loss = torch.cat((pos_inter, neg_inter), 0) #(2*batch ,1)


            #intra_t_loss
            neg_conf = neg_context[:, 1, :]
            neg_author = neg_context[:, 2:, :]
            # fc = list(self.fc_paper.named_parameters())
            # print(fc[0])

            neg_conf = torch.unsqueeze(self.fc_conf(neg_conf), 1)  # batch, 1, outdim
            neg_author = self.fc_author(neg_author)  # batch, len, outdim

            neg_context = torch.cat((neg_conf, neg_author), 1)
            neg_context = self.selfatt(neg_context, entity_type)  # batch, max_len-1, outdim
            neg_context = self.maxpool(neg_context)  # batch, 1, outdim

            neg_t = self.ebl_t(neg_context, paper)
            '''intra_t_neg test'''
            intra_t_neg = neg_t
            neg_t = torch.squeeze(neg_t, -1)

            intra_t_loss = torch.cat((pos_t, neg_t), 0)  # (2*batch,1)


            #intra_loss
            neg_entity_paper = neg_entity[:, 0, :, :]
            neg_entity_conf = neg_entity[:, 1, :, :]
            neg_entity_author = neg_entity[:, 2:, :, :]

            neg_entity_paper = self.fc_paper(neg_entity_paper)  # b, neg_num, dim
            neg_entity_paper = torch.unsqueeze(neg_entity_paper, 1)

            neg_entity_conf = self.fc_conf(neg_entity_conf)  # b, neg_num, dim
            neg_entity_conf = torch.unsqueeze(neg_entity_conf, 1)

            # neg_entity_conf = torch.unsqueeze(self.fc_conf(neg_entity_conf), 1)
            neg_entity_author = self.fc_author(neg_entity_author)  # b, max_len-2, neg_num, dim

            neg_entity = torch.cat((neg_entity_paper, neg_entity_conf, neg_entity_author), 1)
            neg_norm = torch.norm(neg_entity, dim=-1, keepdim=True)  # b, max_len, neg_num, 1
            neg_entity = neg_entity / neg_norm  # b, max_len, neg_num, dim

            pos_matrix = torch.exp(pos_matrix/args.t)
            pos_matrix = torch.mul(pos_matrix, mask)

            for i, sample in enumerate(pos_matrix):
                pos_matrix[i, sample_len[i]:, sample_len[i]:] = 0
            pos_ = torch.sum(pos_matrix, -1)  # (batch, max_len)

            # neg_entity: (batch, max_len, max_neg_len, dim)
            neg_matrix = torch.matmul(torch.unsqueeze(pos_entity, 2),
                                      torch.transpose(neg_entity, 2, 3))  # (batch, max_len, 1, max_neg_len)
            # print(neg_matrix[0][0])
            # neg_matrix = neg_matrix[:, :3, :, :]
            # neg_matrix = self.layer_norm2(torch.squeeze(neg_matrix))
            neg_matrix = torch.squeeze(neg_matrix)
            # neg_matrix = torch.sigmoid(neg_matrix)
            # print("neg1:", neg_matrix[0, :3, :3])
            neg_matrix = torch.exp(neg_matrix/args.t)  # (batch, max_len, max_neg_len)
            neg_ = torch.sum(neg_matrix, -1)  # batch, max_len
            for i, sample in enumerate(neg_):
                neg_[i, sample_len[i]:] = 0

            intra_i = -torch.log(
                (pos_ + 1e-10) / (pos_ + neg_ + 1e-10))  # -torch.log(abs(pos_/(pos_+neg_+1e-8))) #(batch, max_len)

            '''4. max_loss pair-wise test'''
            intra_loss_max = -torch.max(intra_i,-1).values #(batch,) #loss越大,越可能为异常

            intra_loss = torch.unsqueeze(torch.sum(intra_i, -1), -1)  # (batch, 1)
            for i, loss in enumerate(intra_loss):
                intra_loss[i, :] /= sample_len[i]

            return intra_t_loss, intra_loss, inter_loss, inter_neg, intra_t_neg, intra_loss_max

























# if __name__ == "__main__":
#     config = {
#         "num_of_attention_heads": 2,
#         "hidden_size": 4
#     }
#
#     selfattn = SelfAttention(config)
#     print(selfattn)
#     embed_rand = torch.rand((1, 3, 4))
#     print(f"Embed Shape: {embed_rand.shape}")
#     print(f"Embed Values:\n{embed_rand}")
#
#     output = selfattn(embed_rand)
#     print(f"Output Shape: {output.shape}")
#     print(f"Output Values:\n{output}")
