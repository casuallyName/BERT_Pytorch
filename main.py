# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from BERT_Model import *
import pandas





if __name__ == '__main__':
    bert = BERT_Multi_Label_Classification(pre_train_path='./bert-base-chinese', num_labels=8)
    df = pandas.read_csv('train_M.csv').iloc[:20]
    pred_df = pandas.read_csv('train_M.csv').iloc[:20, [0]]
    train_df = df.iloc[:15]
    val_df = df.iloc[15:]
    train_dataloader = Encode_DataFrame(encode_function=bert.encode, dataframe=train_df, batch_size=2, shuffle=True)
    val_dataloader = Encode_DataFrame(encode_function=bert.encode, dataframe=val_df, batch_size=2)
    bert.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=1)
    bert.ROC(val_dataloader=val_dataloader)
#     # bert.save_weight('weight.pkl')
#     # bert.load_weight('weight.pkl')
#     pred_dataloader = Encode_DataFrame(encode_function=bert.encode, dataframe=df, batch_size=2)
#     pred = bert.predict(pred_dataloader=pred_dataloader)
#
#     bert = BERT_Multi_Class_Classification(pre_train_path='bert-base-chinese', num_class=2, from_tf=True)
#     df = pandas.read_excel('test.xlsx').sample(frac=1).iloc[:20]
#     train_df = df.iloc[:int(df.shape[0] * 0.8)]
#     val_df = df.iloc[int(df.shape[0] * 0.8):]
#     train_dataloader = Encode_DataFrame(encode_function=bert.encode, dataframe=train_df, batch_size=2, shuffle=True)
#     val_dataloader = Encode_DataFrame(encode_function=bert.encode, dataframe=val_df, batch_size=2)
#     bert.train(train_dataloader=train_dataloader,val_dataloader=val_dataloader,epochs=1)
#     bert.predict(train_dataloader)



