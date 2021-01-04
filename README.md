# Function

# Model
## BERT 多标签模型
`BERT_Multi_Label_Classification(pre_train_path, num_labels, max_length=128, lr=2e-5, labels_name_list=None, device=None, from_tf=False, output_attentions=False, output_hidden_states=False)`
* Parameters:
    * pre_train_path : ->str, 预训练模型路径
    * num_labels : ->int, 标签数量
    * max_length : ->int, default=128, 句子最大长度
    * lr : ->float, default=2e-5, 学习率 
    * labels_name_list : ->list, default=None, 每个标签的 
    * device : ->str, ('cpu','cuda'), default=None, 使用设备类型
    * from_tf = False
    * output_attentions = False
    * output_hidden_states = False

* Example:
```python
import pandas
from BERT_Model import Encode_DataFrame, BERT_Multi_Label_Classification
bert = BERT_Multi_Label_Classification(pre_train_path='bert-base-chinese', num_labels=2)
train_df = pandas.read_csv('train.csv')
val_df = pandas.read_csv('val.csv')
train_dataloader = Encode_DataFrame(encode_function=bert.encode, dataframe=train_df, batch_size=2, shuffle=True)
val_dataloader = Encode_DataFrame(encode_function=bert.encode, dataframe=val_df, batch_size=2)
bert.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=1)
pred_df = pandas.read_csv('test.csv')
pred_dataloader = Encode_DataFrame(encode_function=bert.encode, dataframe=pred_df, batch_size=2)
pred = bert.predict(pred_dataloader=pred_dataloader)
```
## BERT 多分类（含二分类）模型
`BERT_Multi_Class_Classification(pre_train_path, num_class, max_length=128, lr=2e-5, device=None, from_tf=False, output_attentions=False, output_hidden_states=False)`
* Parameters:
    * pre_train_path : ->str, 预训练模型路径
    * num_class : ->int, 分类数量
    * max_length : ->int, default=128, 句子最大长度
    * lr : ->float, default=2e-5, 学习率
    * device : ->str, ('cpu','cuda'), default=None, 使用设备类型
    * from_tf = False
    * output_attentions = False
    * output_hidden_states = False
    

* Example:
```python
import pandas
from BERT_Model import Encode_DataFrame, BERT_Multi_Class_Classification
bert = BERT_Multi_Class_Classification(pre_train_path='bert-base-chinese', num_class=2)
train_df = pandas.read_csv('train.csv')
val_df = pandas.read_csv('val.csv')
train_dataloader = Encode_DataFrame(encode_function=bert.encode, dataframe=train_df, batch_size=2, shuffle=True)
val_dataloader = Encode_DataFrame(encode_function=bert.encode, dataframe=val_df, batch_size=2)
bert.train(train_dataloader=train_dataloader,val_dataloader=val_dataloader,epochs=1)
pred_df = pandas.read_csv('test.csv')
pred_dataloader = Encode_DataFrame(encode_function=bert.encode, dataframe=pred_df, batch_size=2)
bert.predict(pred_dataloader)
```