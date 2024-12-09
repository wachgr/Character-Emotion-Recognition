
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import BertForPreTraining,BertTokenizer,BertConfig,BertModel
from functools import partial
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW



#加载数据
with open('./data/train_dataset_v2.tsv','r',encoding='utf-8') as handler:
    lines = handler.read().split('\n')[1:-1]
    data = list()
    for line in tqdm(lines):
        # print(line)
        sp = line.split('\t')
        # print(sp)
        if len(sp) != 4:
            print("Error: ", sp)
            continue
        data.append(sp)

train = pd.DataFrame(data)
train.columns = ['id', 'content', 'character', 'emotions']
# print(train)

test = pd.read_csv('./data/test_dataset.tsv', sep='\t')
# print(test)
submit = pd.read_csv('./data/submit_example.tsv', sep='\t')
# print(submit)
train = train[train['emotions'] != '']



#数据处理
train['text'] = train['content'].astype(str) +'角色: ' +train['character'].astype(str)
test['text'] = test['content'].astype(str) +'角色: ' + test['character'].astype(str)
train['emotions'] = train['emotions'].apply(lambda x: [int(_i) for _i in x.split(",")])
# print(train['emotions'])

train[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] = train['emotions'].values.tolist()
test[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] = [0,0,0,0,0,0]
# print(test['love'])
train.to_csv('./data/train.csv',columns=['id', 'content', 'character', 'text', 'love', 'joy', 'fright', 'anger', 'fear', 'sorrow'],sep='\t', index=False)
test.to_csv('./data/test.csv', columns=['id', 'content', 'character', 'text', 'love', 'joy', 'fright', 'anger', 'fear', 'sorrow'],sep='\t', index=False)


#定义dataset
target_cols=['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']
class RoleDataset(Dataset):
    def __init__(self,tokenizer,max_len, mode='train'):
        super(RoleDataset, self).__init__()
        if mode == 'train':
            self.data = pd.read_csv('train.csv',sep='\t')
        else:
            self.data = pd.read_csv('test.csv',sep='\t')
        self.texts=self.data['text'].tolist()
        self.labels=self.data[target_cols].to_dict('records')
        print(self.labels)
        self.tokenizer=tokenizer
        self.max_len=max_len
    def __getitem__(self,index):
        text=str(self.texts[index])
        label=self.labels[index]
        # 这是Tokenizer类的一个方法，用于将文本text转换为模型可以理解的数值表示
        '''
        add_special_tokens=True: 表示在编码的文本前后添加特殊标记（如BERT的[CLS]和[SEP]），这取决于具体的Tokenizer和模型
        max_length=self.max_len: 设置编码后的文本最大长度。如果文本长度超过这个值，将会被截断；如果小于这个值，将会被填充（如果pad_to_max_length为True）。
        pad_to_max_length=True: 如果文本长度小于max_length，则使用填充（通常是0）将其长度增加到max_length。
        return_token_type_ids=True: 对于某些模型（如BERT处理句子对时），需要返回token类型ID来区分不同的句子。这里虽然设置了True，但注意，如果text不是句子对形式，这个信息可能不会被使用。
        return_attention_mask=True: 返回一个注意力掩码，用于指示哪些token是真实的文本，哪些是填充的。这对于模型来说很重要，因为它可以忽略填充部分的计算。
        return_tensors='pt': 指定返回的tensor类型。'pt'代表PyTorch，这意味着返回的encoding字典中的tensor将直接是PyTorch的tensor对象。
        '''
        encoding=self.tokenizer.encode_plus(text,
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            return_token_type_ids=True,
                                            pad_to_max_length=True,
                                            return_attention_mask=True,
                                            return_tensors='pt')
        '''
        'input_ids':encoding['input_ids'].flatten(): 将编码后的输入ID（input_ids）从可能的多维tensor（由于return_tensors='pt'）转换为一维数组。这些ID是模型可以直接处理的数值表示。
        'attention_mask':encoding['attention_mask'].flatten(): 同样，将注意力掩码（attention_mask）从多维tensor转换为一维数组。这个掩码用于指示模型在处理时哪些部分是真实文本，哪些部分是填充。
        '''
        sample={
            'texts':text,
            'input_ids':encoding['input_ids'].flatten(),
            'attention_mask':encoding['attention_mask'].flatten()
        }
        for label_col in target_cols:
            sample[label_col] = torch.tensor(label[label_col],dtype=torch.int64)
        return sample
    def __len__(self):
        return len(self.texts)

'''dataset实现了__getitem__和__len__方法，__getitem__方法用于根据索引获得数据集的单个样本，__len__方法则返回数据集中的样本总数'''
def create_dataloader(dataset,batch_size,mode='train'):
    shuffle = True if mode == 'train' else False
    if mode=='train':
        data_loader=DataLoader(dataset,batch_size=batch_size, shuffle=shuffle)
    else:
        data_loader=DataLoader(dataset,batch_size=batch_size, shuffle=shuffle)
    return data_loader

#加载预训练模型
# roberta
PRE_TRAINED_MODEL_NAME='Untitled Folder'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME,clean_up_tokenization_spaces=True)
base_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)



#模型构建
class EmotionClassifier(nn.Module):
    def __init__(self,n_classes,bert):
        super(EmotionClassifier,self).__init__()
        self.bert = bert
        self.out_love = nn.Linear(self.bert.config.hidden_size,n_classes)
        self.out_joy = nn.Linear(self.bert.config.hidden_size,n_classes)
        self.out_fright = nn.Linear(self.bert.config.hidden_size,n_classes)
        self.out_anger = nn.Linear(self.bert.config.hidden_size,n_classes)
        self.out_fear = nn.Linear(self.bert.config.hidden_size,n_classes)
        self.out_sorrow = nn.Linear(self.bert.config.hidden_size,n_classes)
    def forward(self,input_ids,attention_mask):
        _,pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        love = self.out_love(pooled_output)
        joy = self.out_joy(pooled_output)
        fright = self.out_fright(pooled_output)
        anger = self.out_anger(pooled_output)
        fear = self.out_fear(pooled_output)
        sorrow = self.out_sorrow(pooled_output)
        return  {
            'love':love,'joy':joy,'fright':fright,
            'anger':anger,'fear':fear,'sorrow':sorrow,
        }
EPOCHS=3
weight_decay=0.005
data_path='data'
warmup_proportion=0.0
batch_size=64
lr=2e-5
max_len=128
warm_up_radio=0

trainset = RoleDataset(tokenizer,max_len,mode='train')
train_loader = create_dataloader(trainset,batch_size,mode='train')

valset = RoleDataset(tokenizer,max_len,mode='test')
valid_loader = create_dataloader(valset,batch_size,mode='test')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionClassifier(n_classes=4, bert=base_model)
model.to(device)

#weight_decay指定l2正则化的强度
optimizer = AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
total_steps=len(train_loader)*EPOCHS
#创建一个学习率调度器
scheduler=get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warm_up_radio*total_steps,
    num_training_steps=total_steps
)

#常用的损失函数，用于多分类问题
criterion=nn.CrossEntropyLoss()

#模型训练
def do_train(model,data_loader,criterion,optimizer,scheduler,metric=None):
    model.train()
    global_step=0
    tic_train=time.time()
    log_steps=100
    for epoch in range(EPOCHS):
        losses=[]
        for step,sample in tqdm(enumerate(data_loader)):
            input_ids = sample["input_ids"].to(device)
            attention_mask = sample["attention_mask"].to(device)
            outputs = model(input_ids=input_ids,attention_mask=attention_mask)
            # print(outputs['love'])

            loss_love = criterion(outputs['love'],sample['love'].to(device))
            loss_joy = criterion(outputs['joy'],sample['joy'].to(device))
            loss_fright = criterion(outputs['fright'],sample['fright'].to(device))
            loss_anger = criterion(outputs['anger'],sample['anger'].to(device))
            loss_fear = criterion(outputs['fear'],sample['fear'].to(device))
            loss_sorrow = criterion(outputs['sorrow'],sample['sorrow'].to(device))
            loss=loss_love+loss_joy+loss_fright+loss_anger+loss_fear+loss_sorrow

            losses.append(loss.item())

            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step+=1

            if global_step % log_steps ==0:
                print("global step: %d, epoch :%d, batch:%d, loss:%.5f, speed:%.2f step/s, lr:%.10f"
                      %(global_step, epoch, step, np.mean(losses), global_step/(time.time()-tic_train),
                        float(scheduler.get_last_lr()[0])))


do_train(model,train_loader, criterion, optimizer, scheduler)

#模型预测

from collections import defaultdict
def predict(model,test_loader):
    val_loss=0
    test_pred = defaultdict(list)
    model.eval()
    for step,batch in tqdm(enumerate(test_loader)):
        b_input_ids = batch['input_ids'].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            logists = model(input_ids=b_input_ids,attention_mask=attention_mask)
            for col in target_cols:
                out2 = torch.argmax(logists[col], axis=1)
                test_pred[col].append(out2.cpu().numpy().tolist())

    return test_pred
#加载submit
submit = pd.read_csv('submit_example.tsv',sep='\t')
test_pred = predict(model,valid_loader)

#查看结果
print(test_pred['love'][0])
print(len(test_pred['love']))




label_preds = []

def flatten(arr):
    result = []
    for i in arr:
        if isinstance(i, list):
            result.extend(flatten(i))
        else:
            result.append(i)
    return result

j = 0
for col in target_cols:
    preds = test_pred[col]
    flattened_preds = flatten(preds)
    label_preds.append(flattened_preds)
    print(len(label_preds[j]))
    j += 1
print(len(label_preds))
# print(label_preds)
# print(label_preds[1])
sub = submit.copy()
# 使用 np.column_stack 将它们按列组合在一起
combined_array = np.column_stack((label_preds))
print(combined_array)
sub['emotion']=combined_array.tolist()
print(sub['emotion'])
# sub['emotion'] = np.stack(label_preds,axis=0).tolist()
sub['emotion'] = sub['emotion'].apply(lambda x: ','.join([str(i) for i in x]))
sub.to_csv('baseline_{}.tsv'.format(PRE_TRAINED_MODEL_NAME.split('/')[-1]),sep='\t',index=False)
sub.head()
# 预测结果与输出
# label_preds = []
# for col in target_cols:
#     pred = test_pred[col]
#     label_preds.append(pred)
# print(len(label_preds[0]))
# sub = submit.copy()
# sub['emotion'] = np.stack(label_preds,axis=1).tolist()
# sub['emotion'] = sub['emotion'].apply(lambda x:','.join([str(i) for i in x]))
# sub.to_csv('baseline_{}.tsv'.format(PRE_TRAINED_MODEL_NAME.split('/')[-1]),sep='\t',index=False)
# sub.head()










