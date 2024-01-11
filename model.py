import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.optim as optim
import csv
import random
import numpy as np

from DataPreprocess import cal_distance

class GeoIE_past(nn.Module):
    def __init__(self,user_count,POI_count,emb_dimension,neg_num):
        super(GeoIE_past,self).__init__()
        self.emb_dimension=emb_dimension
        self.scaling=10
        self.negnum=neg_num
        
        #self.a=0.1
        #self.b=-0.2
        self.a = nn.Parameter(torch.FloatTensor(1).uniform_())  # Initialize a
        self.b = nn.Parameter(torch.FloatTensor(1).uniform_())  # Initialize b

        
        self.UserPreference=nn.Embedding(user_count,emb_dimension,sparse=True) # t
        self.PoiPreference=nn.Embedding(POI_count,emb_dimension,sparse=True) # z
        self.GeoInfluence=nn.Embedding(POI_count,emb_dimension,sparse=True) # g
        self.GeoSusceptibility=nn.Embedding(POI_count,emb_dimension,sparse=True) # h
        self.init_emb()

        self.POI_count = POI_count
        self.sigmoid = nn.Sigmoid()

    def init_emb(self):
        nn.init.xavier_normal_(self.UserPreference.weight)
        nn.init.xavier_normal_(self.PoiPreference.weight)
        nn.init.xavier_normal_(self.GeoInfluence.weight)
        nn.init.xavier_normal_(self.GeoSusceptibility.weight)

    def forward(self, cuj, user_id, target, neg_p, History, distance, ng_distance): # neg_p : 5 ng pois of target
        #if cuj > 0:
        wuj = 1 + math.log(1+cuj*(10**self.scaling))
        #else:
        #    wuj = 0
        fij=[] # target,neg distance를 power law에 넣은 값으로 변경
        for d in distance:
            fij.append(self.a.item() * (d ** self.b.item()))
        
        ng_fij = []
        for i in range(len(ng_distance)):
            line=[]
            for d in ng_distance[i]:
                line.append(self.a.item() * (d ** self.b.item()))
            ng_fij.append(line)

        #print(cuj, user_id, target, neg_p, History)
        UPre=self.UserPreference(torch.LongTensor([user_id])) # 1 * emb
        PPre=self.PoiPreference(torch.LongTensor(target)) # 1 * emb
        history_num=len(History) # history num
        loss=[]
        
        hj=self.GeoSusceptibility(torch.LongTensor(target)) # 1 * emb
        g=self.GeoInfluence(torch.LongTensor(History)) # hist_num * emb
        f=torch.FloatTensor(fij)

        yij = (f.mul((hj.mm(g.t())))).sum()
        yij /= float(history_num)
        tz = UPre.mm(PPre.t())
        
        suj= tz + yij
        #posresult = UPre.mm(PPre.t()) + (f.mul((hj.mm(g.t())))).sum()/float(history_num)
        suj= (suj).sigmoid().log()
        loss.append(suj)
        
        for j in range(self.negnum):
            ng_f=torch.FloatTensor(ng_fij[j])
            NegPPre = self.PoiPreference(torch.LongTensor([neg_p[j]])) # neg target, 1 * emb
            Neghj=self.GeoSusceptibility(torch.LongTensor([neg_p[j]])) # 1 * emb
            
            ng_yij = (ng_f.mul((Neghj.mm(g.t())))).sum()
            ng_yij /= float(history_num)
            ng_tz = UPre.mm(NegPPre.t())
            
            ng_suj= ng_tz + ng_yij
            ng_suj= (1-(ng_suj.sigmoid())).log()
            loss.append(ng_suj)
        L = wuj * sum(loss)
        return -L #SGA를 위해 부호 전환


    def predict(self, user_id, History, POI_info):
        targets = [i for i in range(self.POI_count) if i not in History]
        predict = []

        for target in targets:
            # if target in History:
            #     continue
            
            # distance list 생성, target에 대하여 history 속 모든 poi와의 거리 계산
            distance = cal_distance(target,History,POI_info)

            fij=[] # target,neg distance를 power law에 넣은 값으로 변경
            for i in distance:
                tmp=self.a * i ** self.b
                fij.append(tmp)

            UPre=self.UserPreference(torch.LongTensor([user_id])) # 1 * emb
            PPre=self.PoiPreference(torch.LongTensor([target])) # 1 * emb

            history_num=len(History) # history num

            hj=self.GeoSusceptibility(torch.LongTensor([target])) # 1 * emb
            g=self.GeoInfluence(torch.LongTensor(History)) # hist_num * emb
            f=torch.FloatTensor([fij]) # 1 * hist_num
            
            yij = (f.mul((hj.mm(g.t())))).sum()
            yij /= float(history_num)
            tz = UPre.mm(PPre.t())
            
            suj = tz + yij
            puj = F.softmax(suj, dim = 1)
            #print(target, posresult)
            predict.append([target, puj.item()])

        return predict
    
class testmodel(nn.Module):
    def __init__(self,user_count,POI_count,emb_dimension):
        super(testmodel,self).__init__()
        self.emb_dimension=emb_dimension
        self.scaling=10
        self.a=0.1
        self.b=-0.2
        self.UserPreference=nn.Embedding(user_count,emb_dimension,sparse=True)
        self.PoiPreference=nn.Embedding(POI_count,emb_dimension,sparse=True)
        self.GeoInfluence=nn.Embedding(POI_count,emb_dimension,sparse=True)
        self.GeoSusceptibility=nn.Embedding(POI_count,emb_dimension,sparse=True)
        self.init_emb()
    def init_emb(self):
        nn.init.xavier_normal(self.UserPreference.weight)
        nn.init.xavier_normal(self.PoiPreference.weight)
        nn.init.xavier_normal(self.GeoInfluence.weight)
        nn.init.xavier_normal(self.GeoSusceptibility.weight)
    def forward(self,user):
        emb_u=self.UserPreference(torch.LongTensor(user))
        list=torch.mm(self.PoiPreference.weight,emb_u.t())
        list=list.view(-1)
        return list
    

class GeoIE(nn.Module):
    def __init__(self,user_count,POI_count,emb_dimension,neg_num,a,b):
        super(GeoIE,self).__init__()
        self.emb_dimension=emb_dimension
        self.scaling=10
        self.negnum=neg_num
        self.a=a
        self.b=b
        # self.a = nn.Parameter(torch.FloatTensor(1).uniform_(-1,1))  # Initialize a
        # self.b = nn.Parameter(torch.FloatTensor(1).uniform_(-1,1))  # Initialize b

        
        self.UserPreference=nn.Embedding(user_count,emb_dimension) # t
        self.PoiPreference=nn.Embedding(POI_count,emb_dimension) # z
        self.GeoInfluence=nn.Embedding(POI_count,emb_dimension) # g
        self.GeoSusceptibility=nn.Embedding(POI_count,emb_dimension) # h
        self.init_emb()

        self.POI_count = POI_count
        self.sigmoid = nn.Sigmoid()
        self.loss_func = nn.BCELoss()

    def init_emb(self):
        nn.init.xavier_normal_(self.UserPreference.weight)
        nn.init.xavier_normal_(self.PoiPreference.weight)
        nn.init.xavier_normal_(self.GeoInfluence.weight)
        nn.init.xavier_normal_(self.GeoSusceptibility.weight)

    def forward(self, user_id, targets, history, check_in_num, distances):
        # user_id = torch.LongTensor(user_id)  #
        # targets = torch.LongTensor(targets) # 
        # history = torch.LongTensor(history) # 
        cuj = check_in_num # len : target

        UPre = self.UserPreference(user_id) #b * emb 
        PPre = self.PoiPreference(targets) #b * emb 
        hj = self.GeoSusceptibility(targets) #b * emb 

        g = self.GeoInfluence(history) #b * h * emb 
        history_size = len(history[0])
        batch_size = len(history)
        g = g.reshape([batch_size,-1,history_size]) # b * emb * h
        fij = self.a * (distances**self.b)
        hj = torch.reshape(hj,[batch_size,-1,1]) #b * emb * 1
        t1 = g*hj  # b * emb * h
        t2 = torch.sum(t1,dim=1) * fij # b * h
        yij = torch.sum(t2,dim=-1) / history_size 

        UPre = torch.reshape(UPre,[batch_size,1,-1])
        PPre = torch.reshape(PPre,[batch_size,-1,1])

        tz = torch.bmm(UPre,PPre).squeeze(-1) # t * 1
        
        suj = tz + yij.unsqueeze(1) # t * 1

        wuj = 1 + torch.log(1+cuj*(10**self.scaling))
        return self.sigmoid(suj), wuj

    
    def loss_function(self, prediction, label, weight):
        t1 = label * (torch.log(prediction+1e-10))
        t2 = (1-label)*(torch.log(1-prediction+1e-10))
        loss = -weight * (t1+t2)
        loss = torch.sum(loss)
        temp = torch.isnan(loss)
        if temp.item() == True:
            print(prediction)
            print(label)
            print(weight)
            print(t1)
            print(t2)
        return torch.sum(loss)

