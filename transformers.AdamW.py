from transformers import AdamW,get_linear_schedule_with_warmup
from torch.optim import Optimizer
import torch
from torch.utils.data import DataLoader,TensorDataset
from transformers import BertModel,BertTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

texts = ["Hello, world!","BERT is great for NLP tasks"]
labels= [0,1]

#对我文本进行编码
inputs = tokenizer(texts,padding=True, truncation=True,return_tensors="pt")

dataset = TensorDataset(inputs['input_ids'],inputs['attention_mask'],torch.tensor(label))
dataloader = DataLoader(dataset,batch_size=2,shuffle=True)

optimizer = AdamW(model.parameters(),lr=5e-5)
num_warmup_steps =10
num_training_steps=100

#创建一个学习率调度器
scheduler=get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

model.train()
num_epoch =3
#训练循环
for epoch in range(num_epoch):
    for batch in dataloader:
        batch = {k: v.to(device) for k,v in batch.items}
        #前向传播
        outputs = model(**batch, labels=None)
        optimizer.step()
        scheduler.step()
        #清除梯度
        optimizer.zero_grad()