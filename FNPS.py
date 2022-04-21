 # * **************************************************************************
 # * ********************                                  ********************
 # * ********************      COPYRIGHT INFORMATION       ********************
 # * ********************                                  ********************
 # * **************************************************************************
 # *                                                                          *
 # *                                   _oo8oo_                                *
 # *                                  o8888888o                               *
 # *                                  88" . "88                               *
 # *                                  (| -_- |)                               *
 # *                                  0\  =  /0                               *
 # *                                ___/'==='\___                             *
 # *                              .' \\|     |// '.                           *
 # *                             / \\|||  :  |||// \                          *
 # *                            / _||||| -:- |||||_ \                         *
 # *                           |   | \\\  -  /// |   |                        *
 # *                           | \_|  ''\---/''  |_/ |                        *
 # *                           \  .-\__  '-'  __/-.  /                        *
 # *                         ___'. .'  /--.--\  '. .'___                      *
 # *                      ."" '<  '.___\_<|>_/___.'  >' "".                   *
 # *                     | | :  `- \`.:`\ _ /`:.`/ -`  : | |                  *
 # *                     \  \ `-.   \_ __\ /__ _/   .-` /  /                  *
 # *                 =====`-.____`.___ \_____/ ___.`____.-`=====              *
 # *                                   `=---=`                                *
 # * **************************************************************************
 # * ********************                                  ********************
 # * ********************                                  ********************
 # * ********************         佛祖保佑 永远无BUG         ********************
 # * ********************                                  ********************
 # * **************************************************************************

import torch
import ImageFeature
import TextFeature
import FuseAllFeature
from LoadData import *
from torch.utils.data import Dataset, DataLoader,random_split
import numpy as np
import os
import time
import itertools
from visdom import Visdom
import sklearn.metrics as metrics
import seaborn as sns
import Net
from torch.autograd import Variable, Function
import torch.nn.functional as F
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#训练过程可视化
loss = Visdom()
acc=Visdom()
loss.line([[0,0]],[1],win='train_loss', opts=dict(title='loss', legend=['siamese_loss','domain_loss']))
acc.line([[0,0]],[1],win='train_acc', opts=dict(title='acc', legend=['siamese_acc','domain_acc']))

def to_np(x):
    return x.data.cpu().numpy()
def create_folder(foldermodal_names):
    current_position = "./model_sava/"
    foldermodal_name=str(current_position)+str(foldermodal_names)+"/"
    isCreate=os.path.exists(foldermodal_name)
    if not isCreate:
        os.makedirs(foldermodal_name)
        print(str(foldermodal_name)+'is created')
    else:
        print('Already exist')
        return False

# def recoder(modal_path,text):
# 	with open(modal_path+'recoder.txt','a',encoding='utf-8') as f:
# 		f.write(text+'\n')
# 		f.close()

def model_load():
	checkpoint1 = torch.load('./model_sava/extractor.pth')
	checkpoint2 = torch.load('./model_sava/siamese.pth')
	checkpoint3 = torch.load('./model_sava/domain_classifier.pth')
	extractor.load_state_dict(checkpoint1['model'])
	siamese.load_state_dict(checkpoint2['model'])
	domain_classifier.load_state_dict(checkpoint3['model'])
	optimizer_extractor.load_state_dict(checkpoint1['optimizer'])
	optimizer_siamese.load_state_dict(checkpoint2['optimizer'])
	optimizer_domain.load_state_dict(checkpoint3['optimizer'])
	print('Model_Load completed\n')
	return extractor,siamese,domain_classifier,optimizer_extractor,optimizer_siamese,optimizer_domain

def train(extractor,siamese,domain_classifier,optimizer_extractor,optimizer_siamese,
		optimizer_domain,siamese_loss_fn,train_loader,number_of_epoch,is_train):
	step=0
	for epoch in range(number_of_epoch):
		siamese_loss_sum=domain_loss_sum=siamese_correct_num=domain_correct_num=data_num=0
		extractor.train()
		siamese.train()
		domain_classifier.train()
		batch=0
		for text_index,image_feature,lable,domain,id in train_loader:
			step+=1
			batch+=1
			batch_num=len(train_loader)
			data_num+=lable.shape[0]
			lable = lable.view(-1,1).to(torch.float32).to(device)
			domain = domain.to(device)

			text_feature,image_feature=extractor(text_index.to(device),image_feature.to(device))
			output1,output2=siamese(text_feature,image_feature)
			domain_feature=torch.cat((text_feature,image_feature),1)

			domain_pred=domain_classifier(domain_feature)
			distance=F.pairwise_distance(output1, output2)
			one  = torch.ones_like(distance)
			zero  = torch.zeros_like(distance)
			siamese_pred=torch.where(distance<=0.65,zero,one).round().view(-1,1)

			siamese_loss = siamese_loss_fn(output1,output2,lable)
			domain_loss=domain_loss_fn(domain_pred,domain)
			siamese_loss_sum += siamese_loss.item()
			domain_loss_sum += domain_loss
			siamese_correct_num+=(siamese_pred==lable).sum().item()
			domain_correct_num += (domain_pred.argmax(dim=1)==domain).sum().item()

			optimizer_extractor.zero_grad()
			optimizer_siamese.zero_grad()
			optimizer_domain.zero_grad()

			siamese_loss.backward(retain_graph=True)
			domain_loss.backward(retain_graph=False)

			if is_train==1:
				optimizer_domain.step()
				optimizer_siamese.step()
				optimizer_extractor.step()
			else:
				pass

			print('Training...epoch:%d/%d'%(epoch+1,number_of_epoch))
			print('batch:%d/%d'%(batch,batch_num))
			print("siamese:train_loss=%.5f train_acc=%.4f"%(siamese_loss_sum/data_num,siamese_correct_num/data_num))
			print("domain:train_loss=%.5f train_acc=%.4f"%(domain_loss_sum/data_num,domain_correct_num/data_num))
			print('')
#chart
			loss.line([[siamese_loss_sum/data_num,domain_loss_sum.item()/data_num]], 
				[step], win='train_loss', update='append')
			acc.line([[siamese_correct_num/data_num,domain_correct_num/data_num]], 
				[step], win='train_acc', update='append')

			if batch==1:
				lable_list = list(to_np(lable.squeeze()))
				pred_list = list(to_np(siamese_pred.round().squeeze()))
			else:
				lable_list=lable_list+list(to_np(lable.squeeze()))
				pred_list=pred_list+list(to_np(siamese_pred.round().squeeze()))

		precision=precision_score(lable_list,pred_list)
		recall=recall_score(lable_list,pred_list)
		f1=f1_score(lable_list,pred_list)
		print('Precision：'+str(precision))
		print('Recall：'+str(recall))
		print('F1-score：'+str(f1))
#sava model
		state1 = {'model': extractor.state_dict(), 'optimizer': optimizer_extractor.state_dict()}
		state2 = {'model': siamese.state_dict(), 'optimizer': optimizer_siamese.state_dict()}
		state3 = {'model': domain_classifier.state_dict(), 'optimizer': optimizer_domain.state_dict()}
		torch.save(state1, './model_sava/extractor.pth')
		torch.save(state2, './model_sava/siamese.pth')
		torch.save(state3, './model_sava/domain_classifier.pth')

learning_rate = 0.001
lstm_dropout_rate = 0
weight_decay = 0
train_fraction=0.9
val_fraction=0.1
batch_size=64   #############
data_shuffle=True
number_of_epoch=5   #############
train_data=my_data_set(train_data_set)
test_set=my_data_set(test_data_set)
train_set,val_set=train_val_split(train_data,train_fraction,val_fraction)
# load data
train_loader = DataLoader(train_set,batch_size=batch_size, shuffle=data_shuffle)
val_loader = DataLoader(val_set,batch_size=batch_size, shuffle=data_shuffle)
test_loader = DataLoader(test_set,batch_size=batch_size, shuffle=data_shuffle)
# start train
if __name__ == "__main__":
# loss function
	siamese_loss_fn = Net.ContrastiveLoss()
	domain_loss_fn = torch.nn.CrossEntropyLoss()
# initilize the model
	extractor=Net.FeatureExtractor(lstm_dropout_rate).to(device)
	siamese=Net.Siamese().to(device)
	domain_classifier=Net.DomainClassifier().to(device)
# optimizer
	optimizer_extractor = torch.optim.Adam(extractor.parameters(), lr=learning_rate,weight_decay=weight_decay)
	optimizer_siamese = torch.optim.Adam(siamese.parameters(), lr=learning_rate,weight_decay=weight_decay)
	optimizer_domain = torch.optim.Adam(domain_classifier.parameters(), lr=learning_rate,weight_decay=weight_decay)
#model load
	modal_path='./model_sava/'
	if os.listdir('./model_sava/')!=[]:
		model_load()
# train
	train(extractor,siamese,domain_classifier,optimizer_extractor,optimizer_siamese,
		optimizer_domain,siamese_loss_fn,test_loader,number_of_epoch,11)

