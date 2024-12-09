from transformers import BertTokenizer,BertConfig,BertModel,BertForPreTraining

#加载预训练的分词器
tokenier = BertTokenizer.from_pretrained('bert-base-uncased')

#加载预训练的模型配置
config = BertConfig.from_pretrained('bert-base-uncased')

#加载预训练的BERT模型（基本模型，不包含预训练任务特定的层）
model = BertModel.from_pretrained('bert-base-uncased')

#加载预训练的BERT预训练的模型（包含MLM和NSP任务的层）
pretain_model=BertForPreTraining.from_pretrained('bert-base-uncased')