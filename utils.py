# coding: utf-8
import random
import numpy as np
import scipy.io as sio
import scipy.sparse as sp



#load triples from file
class init:
    def __init__(self):
        pass


    def getAllPosNegEvent(self, path):
        d = np.load(path, allow_pickle=True).item()
        neg_event = d['neg_event']
        pos_event = d['pos_event']
        return pos_event, neg_event

    def saveAllPosNegEvent(self, path):
        pos_dict = {}
        neg_dict = {}

        entity_list = []
        with open(path, "r") as fp:
            temp = fp.readline()
            for k in fp.readlines():
                entity_set=set(k.strip().split('\t')[0:-2])
                entity_list.append(entity_set)
        pos_len = []
        neg_len = []
        for i, k1 in enumerate(entity_list):
            max=0
            min=10
            for j, k2 in enumerate(entity_list):
                if(j!=i):
                    l = len(k1&k2)
                    if(l>max):
                        max=l
                    if(l<min):
                        min=l
            pos_len.append(max)
            neg_len.append(min)
        for i, k1 in enumerate(entity_list):
            pos_dict[i] = set()
            neg_dict[i] = set()
            for j, k2 in enumerate(entity_list):
                if(j!=i):
                    l = len(k1&k2)
                    if(pos_len[i]==l):
                        pos_dict[i].add(j)
                    if(neg_len[i]==l):
                        neg_dict[i].add(j)
        # print(pos_dict)
        # print(neg_dict)
        for i, k in neg_dict.items():
            s = random.sample(neg_dict[i], 1000)
            neg_dict[i] = set(s)

        np.save('./data_event/posneg_event.npy', {'neg_event': neg_dict, 'pos_event': pos_dict})


    #neg_entity
    def getAllNeg(self, path):
        d = np.load(path, allow_pickle=True).item()
        conf_neg_set = d['conf_neg_set']
        author_neg_set = d['author_neg_set']
        paper_neg_set = d['paper_neg_set']
        return paper_neg_set, conf_neg_set, author_neg_set


    def saveAllNeg(self, path):
        paper_set = set()
        conf_set = set()
        author_set = set()
        paper_pos_set = {}
        conf_pos_set = {}
        author_pos_set = {}
        paper_neg_set = {}
        conf_neg_set = {}
        author_neg_set = {}
        with open(path, 'r') as fp:
            line = fp.readline()
            for line in fp.readlines():
                l = line.strip().split('\t')
                paper = int(l[0])
                conf = int(l[1])
                author = l[2:-2]
                paper_set.add(paper)
                conf_set.add(conf)
                paper_pos_set[paper] = set()
                paper_pos_set[paper].add(conf)
                if(conf not in conf_pos_set.keys()):
                    conf_pos_set[conf] = set()
                conf_pos_set[conf].add(paper)
                for a in author:
                    ai = int(a)
                    author_set.add(ai)
                    paper_pos_set[paper].add(ai)
                    conf_pos_set[conf].add(ai)
                    if(ai not in author_pos_set.keys()):
                        author_pos_set[ai] = set()
                    for a2 in author:
                        if(int(a2)!=ai):
                            author_pos_set[ai].add(int(a2))
        print("author:", len(author_pos_set))
        print("conf:", len(conf_pos_set))
        print("paper:", len(paper_pos_set))

        for key in paper_pos_set.keys():
            paper_neg_set[key] = set()
        for key in author_pos_set.keys():
            author_neg_set[key] = set()
        for key in conf_pos_set.keys():
            conf_neg_set[key] = set()

        a_c = author_set | conf_set
        p_a = paper_set | author_set
        p_a_c = paper_set | conf_set | author_set
        # print(len(a_c))
        # print(len(p_a))
        # print(len(p_a_c))

        for k in paper_neg_set.keys():
            for i in a_c:
                if(i not in paper_pos_set[k]):
                    paper_neg_set[k].add(i)

        max = 0
        min = 10000000
        for i in paper_neg_set.keys():
            temp = random.sample(paper_neg_set[i], 1000)
            paper_neg_set[i] = temp
            if(len(paper_neg_set[i])>max):
                max = len(paper_neg_set[i])
            if (len(paper_neg_set[i]) < min):
                min = len(paper_neg_set[i])
        print(min, max)

        for k in author_neg_set.keys():
            for i in p_a_c:
                if(i not in author_pos_set[k]):
                    author_neg_set[k].add(i)
        max = 0
        min = 10000000
        for i in author_neg_set.keys():
            temp = random.sample(author_neg_set[i], 1000)
            author_neg_set[i] = temp
            if (len(author_neg_set[i]) > max):
                max = len(author_neg_set[i])
            if (len(author_neg_set[i]) < min):
                min = len(author_neg_set[i])
        print(min, max)

        for k in conf_neg_set.keys():
            for i in p_a:
                if(i not in conf_pos_set[k]):
                    conf_neg_set[k].add(i)
        max = 0
        min = 10000000
        for i in conf_neg_set.keys():
            temp = random.sample(conf_neg_set[i], 1000)
            conf_neg_set[i] = temp
            if (len(conf_neg_set[i]) > max):
                max = len(conf_neg_set[i])
            if (len(conf_neg_set[i]) < min):
                min = len(conf_neg_set[i])
        print(min, max)

        np.save('./data_event/neg_entity.npy', {'conf_neg_set': conf_neg_set, 'paper_neg_set': paper_neg_set, \
                                                                                'author_neg_set': author_neg_set})



    def getSamples(self, path):
        sample_list = []
        length = []
        with open(path, 'r') as fp:
            line = fp.readline()
            for line in fp.readlines():
                sample = [int(id) for id in line.strip().split('\t')[:-2]]
                len = [int(id) for id in line.strip().split('\t')[-2:]]
                sample_list.append(sample)
                length.append(len)
        return sample_list, length

    def getNodeMapId(self, id_map_path, entity_type):
        map = np.load(id_map_path, allow_pickle=True).item()
        entity_map = map[entity_type+'_map']
        return entity_map



    def getNodeFeatures(self, path):
        with open(path, 'r') as f:  # mp1是全部的node（16万）的feature
            feature_dic = {}  # key:node_id, value:feature(108 dim)
            n = 0
            for line in f:
                line = line.strip().split('\t')[1].split(' ')
                ll = []
                for l in line:
                    try:
                        l = float(l)
                    except:
                        l = 0.0
                    ll.append(l)
                feature_dic[n] = ll
                n += 1
        return feature_dic

    def getEntityLabel(self, path): #author, conf, paper
        with open(path, 'r') as f:
            label_entity = {}  # dict 键为领域（1，2.。）值为一个列表，表示这个领域的author列表
            entity_label = {}  # dict author的label（领域1，2...）
            for line in f:
                line = line.strip().split('\t')
                entity, label = int(line[0]), int(line[1])
                entity_label[entity] = label
                if label not in label_entity.keys():
                    label_entity[label] = []
                label_entity[label].append(entity)

        return entity_label, label_entity

    def generate_samples(self, sample_list, length, max_len):

        max_len = max_len-1#except paper
        #gen entity_type
        size = len(sample_list)
        entity_type = np.zeros([size, max_len])
        for i, le in enumerate(length):
            entity_type[i, :le[0]] = 1
            entity_type[i, le[0]:le[0]+le[1]] = 2

        return np.array(sample_list, dtype=object), entity_type




    def generate_neg_samples(self, sample_list, entity_label, all_neg_set, all_event_set, neg_sum):
        # (data, author2paperId, paperId2venueId, paperId2authorId, authorId2paperId,  venueId2paperId, author2id, a_c,
        #                                                                                 p_a, p_a_c, id2id) = all_data
        (author_label, paper_label, conf_label, label_author, label_paper, label_conf) = entity_label
        (paper_neg_set, conf_neg_set, author_neg_set) = all_neg_set
        (event_pos_set, event_neg_set) = all_event_set
        pos_event = []
        neg_event = []
        neg_context = []
        neg_entity = []
        for i, sample in enumerate(sample_list):
            #pos_event
            random_pos_event_id = random.sample(list(event_pos_set[i]), 1)[0]
            random_pos_event = sample_list[random_pos_event_id]
            pos_event.append(random_pos_event)
            #neg_event
            random_neg_event_id = random.sample(list(event_neg_set[i]), 1)[0]
            random_neg_event = sample_list[random_neg_event_id]
            neg_event.append(random_neg_event)


            # neg context
            #1. neg paper
            # cur_paper_label = paper_label[sample[0]]
            # random_paper_cate = random.sample(label_paper.keys, 1)
            # while(random_paper_cate == cur_paper_label):
            #     random_paper_cate = random.sample(label_paper.keys, 1)
            # neg_paper = random.sample(label_paper[random_paper_cate], 1)

            #2. neg conf
            cur_conf = sample[1]
            cur_conf_label = conf_label[cur_conf]
            random_conf_label = random.sample(label_conf.keys(), 1)[0]
            while (random_conf_label == cur_conf_label):
                random_conf_label = random.sample(label_conf.keys(), 1)[0]
            neg_conf = random.sample(label_conf[random_conf_label], 1)[0]

            #3. neg_author
            cur_author_list = sample[2:]
            cur_author = random.sample(cur_author_list, 1)[0]
            cur_author_label = author_label[cur_author]
            random_author_label = random.sample(label_author.keys(), 1)[0]
            neg_author = random.sample(label_author[random_author_label], 1)[0]
            while (random_author_label == cur_author_label or neg_author in cur_author_list):
                random_author_label = random.sample(label_author.keys(), 1)[0]
                neg_author = random.sample(label_author[random_author_label], 1)[0]
            replace_index = cur_author_list.index(cur_author)
            cur_author_list[replace_index] = neg_author

            neg_context.append([sample[0]]+[neg_conf]+cur_author_list)


            #neg_entity
            cur_neg_entity = []

            #1. paper
            cur_paper = sample[0]
            random_p_n = random.sample(list(paper_neg_set[cur_paper]), neg_sum)
            cur_neg_entity.append(random_p_n)

            #2. conf
            cur_conf = sample[1]
            random_c_n = random.sample(list(conf_neg_set[cur_conf]), neg_sum)
            cur_neg_entity.append(random_c_n)

            #3. author
            cur_author_list = sample[2:]
            for author in cur_author_list:
                random_c_n = random.sample(list(author_neg_set[author]), neg_sum)
                cur_neg_entity.append(random_c_n)

            neg_entity.append(cur_neg_entity) #(batch, len, 10)



        return np.array(neg_context, dtype=object), np.array(neg_entity, dtype=object), np.array(pos_event, dtype=object),\
               np.array(neg_event, dtype=object)



    def get_features(self, events, features, max_len, neg_sum):
        #conf:35, paper:108, author:58
        return_features = []
        batch_size = len(events)
        #print(events.ndim)
        if(len(np.shape(events[0]))==1):#event
            return_features = np.zeros([batch_size, max_len, 108])
            for i, sample in enumerate(events):
                for j, index in enumerate(sample):
                    cur_feature = features[index]
                    cur_feature = np.array(cur_feature, dtype='float')
                    cur_feature = np.concatenate((cur_feature, np.zeros([108-len(cur_feature)])), 0)
                    return_features[i, j] = cur_feature
        if(len(np.shape(events[0]))==2): #neg_entity
            return_features = np.zeros([batch_size, max_len, neg_sum, 108])
            for i, sample in enumerate(events):
                for j, index1 in enumerate(sample):
                    for k, index2 in enumerate(index1):
                        cur_feature = features[index2]
                        cur_feature = np.array(cur_feature, dtype='float')
                        cur_feature = np.concatenate((cur_feature, np.zeros([108 - len(cur_feature)])), 0)
                        return_features[i, j, k] = cur_feature


        return return_features #[batch_size, max_len, 108]










