# This code is used as template for meta-learning
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import utils
from abc import abstractmethod

class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way = False):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = 15 #(change depends on input) 
        self.feature    = model_func()
        self.feat_dim   = self.feature.final_feat_dim
        self.change_way = change_way  #some methods allow different_way classification during training and test

    @abstractmethod
    def set_forward(self,x,is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self,x):
        x = x.permute(0,2,1)
        out  = self.feature.forward(x)
        return out

    def parse_feature(self,x,is_feature):
        x    = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            if x.size()[0] != self.n_way * (self.n_support + self.n_query):
                x = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
            #print("DDDDDDDDDDDDDDD",x.shape)
            x           = x.permute(0,2,1)
            z_all       = self.feature.forward(x)
            z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        ## All additional modules will be added at this point
        from mymodules import FewShotPointCloudLearner
        from io_utils import parse_args
        params = parse_args('train')
        learner = FewShotPointCloudLearner(z_query, z_proto,params)
        z_query, z_proto = learner.self_interaction_and_attention(z_query, z_proto)
        return z_proto,z_query




    def correct(self, x):       
        scores = self.set_forward(x)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer ):
        
        print_freq = 1000 # just show epochs as its crearing disturbance on tmux

        avg_loss=0
        for i, (x,_ ) in enumerate(train_loader):
            #print("####",i,x.shape)            
            # self.n_query = x.size(1) - self.n_support           
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss( x )
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.item()

            if i % print_freq==0:
                print('Epoch {:d} | Episode {:d} | Loss {:f}'.format(epoch, i, avg_loss/float(i+1)))
        return avg_loss/float(i+1)
        #return avg_loss
    

    def test_loop(self, test_loader, record = None):
        correct =0
        count = 0
        acc_all = []
        
        # iter_num = len(test_loader) 
        from io_utils import parse_args
        params = parse_args('train')
        iter_num = params.iter_num_test

        for i, (x,_) in enumerate(test_loader):
            # self.n_query = x.size(1) - self.n_support
            self.n_query = 15
            if self.change_way:
                self.n_way  = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/ count_this*100  )

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        return acc_mean


    def set_forward_adaptation(self, x, is_feature=True):
        assert is_feature == True, 'Feature is fixed in further adaptation'
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        y_support = Variable(y_support.cuda())

        linear_clf = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim, bias=True),  # Adding an intermediate layer
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.n_way, bias=True)

        )
        linear_clf = linear_clf.cuda()

        optimizer = torch.optim.Adam(linear_clf.parameters(), lr=0.01, weight_decay=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)  # Learning rate scheduler

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()

        epochs = 100
        for epoch in range(epochs):
            linear_clf.train()
            lr_scheduler.step()  # Update learning rate

            for i in range(0, len(y_support), self.n_support):
                optimizer.zero_grad()
                z_batch = z_support[i:i + self.n_support]
                y_batch = y_support[i:i + self.n_support]
                scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)
                loss.backward()
                optimizer.step()

        linear_clf.eval()
        scores = linear_clf(z_query)
        return scores


    def set_forward_adaptationold(self, x, is_feature = True): #further adaptation, default is fixing feature and train a new softmax clasifier
        assert is_feature == True, 'Feature is fixed in further adaptation'
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.cuda())

        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()
        
        batch_size = 1
        support_size = self.n_way* self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id] 
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()
        scores = linear_clf(z_query)
        return scores
