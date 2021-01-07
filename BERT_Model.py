#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/25 9:56
# @Author  : ZhouHang
# @Email   : zhouhang@idataway.com
# @File    : BERT_Model.py
# @Software: PyCharm

import os
import numpy
import torch
import pandas
import warnings
import datetime
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

__all__ = ['Encode_DataFrame', 'BERT_Multi_Class_Classification', 'BERT_Multi_Label_Classification']


def Encode_DataFrame(encode_function, dataframe, batch_size, shuffle=False):
    if dataframe.shape[1] == 2:
        text_values = dataframe.iloc[:, 0].values
        labels_values = dataframe.iloc[:, 1:].values
        labels_values = torch.tensor(labels_values)
    else:
        text_values = dataframe.iloc[:, 0].values
        labels_values = dataframe.iloc[:, 1:].values
        labels_values = torch.Tensor(labels_values)

    text_values = encode_function(text_values)
    dataset = TensorDataset(text_values, labels_values)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class AverageMeter(object):
    def __init__(self):
        self.count = 0
        self.val = 0
        self.sum = 0
        self.avg = 0

    def __str__(self):
        return '{:<0.3f}'.format(round(self.avg, 3))

    def reset(self):
        self.count = 0
        self.sum = 0
        self.val = 0
        self.avg = 0

    def update(self, val, n=1):
        self.count += n
        self.val = val
        self.sum += self.val
        self.avg = self.sum / self.count
        return '{:<0.3f}'.format(round(self.avg, 3))


class BERT_Multi_Label_Classification(object):

    def __init__(self, pre_train_path, num_labels, max_length=128, lr=2e-5, labels_name_list=None, device=None,
                 from_tf=False, output_attentions=False, output_hidden_states=False):

        self.__tokenizer = BertTokenizer.from_pretrained(pre_train_path, do_lower_case=True)

        self.model = BertForSequenceClassification.from_pretrained(pre_train_path, num_labels=num_labels,
                                                                   output_attentions=output_attentions,
                                                                   output_hidden_states=output_hidden_states,
                                                                   from_tf=from_tf)

        self.lr = lr
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.BCELoss()
        self._num_labels = num_labels
        self._Sigmoid = torch.nn.Sigmoid()
        self._max_length = max_length
        self._Train_Loss = AverageMeter()
        self._Train_Accuracy = AverageMeter()
        self._Val_Loss = AverageMeter()
        self._Val_Accuracy = AverageMeter()
        self._global_epoch = 0
        if labels_name_list is not None:
            self.labels_name_list = labels_name_list
        else:
            self.labels_name_list = [f'label_{i}' for i in range(self._num_labels)]
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
        else:
            self.device = torch.device(device)
            if device == 'cuda':
                self.device = torch.device('cuda')
            elif device == 'cpu':
                self.device = torch.device('cpu')
                if torch.cuda.is_available():
                    warnings.warn("cuda is available, you can set (device='cuda') to speed up the program",
                                  RuntimeWarning)
            else:
                raise ValueError("device has to be one of (None, 'cpu', 'cuda')")
            self.model.to(self.device)

    def __str__(self):
        return str(self.model)

    def __save_checkpoint(self, path):
        file_name = datetime.datetime.now().strftime('checkpoint_%Y_%m_%d_%H_%M_%S.pth.tar')
        torch.save({'epoch': self._global_epoch, 'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, os.path.join(path, file_name))

    def __load_checkpoint(self, path):
        model_checkpoint_dict = torch.load(path)
        self._global_epoch = model_checkpoint_dict['epoch'] - 1
        start = model_checkpoint_dict['epoch']
        self.model.load_state_dict(model_checkpoint_dict['state_dict'])
        self.optimizer.load_state_dict(model_checkpoint_dict['optimizer'])
        return start

    def __calculate_metrics(self, y_true, y_pred, threshold=0.5):
        y_pred = numpy.array(y_pred > threshold, dtype=float)
        warnings.filterwarnings('ignore')
        metrics = {
            'micro': {
                'precision': precision_score(y_true=y_true, y_pred=y_pred, average='micro'),
                'recall': recall_score(y_true=y_true, y_pred=y_pred, average='micro'),
                'F1': f1_score(y_true=y_true, y_pred=y_pred, average='micro')
            },
            'macro': {
                'precision': precision_score(y_true=y_true, y_pred=y_pred, average='macro'),
                'recall': recall_score(y_true=y_true, y_pred=y_pred, average='macro'),
                'F1': f1_score(y_true=y_true, y_pred=y_pred, average='macro')
            },
            'samples': {
                'precision': precision_score(y_true=y_true, y_pred=y_pred, average='samples'),
                'recall': recall_score(y_true=y_true, y_pred=y_pred, average='samples'),
                'F1': f1_score(y_true=y_true, y_pred=y_pred, average='samples')
            }
        }
        return metrics

    def encode(self, text_list):
        '''
        用来把文本转换成token并进行pading

        :param text_list:
        :return:
        '''
        input_ids_list = []
        for text in tqdm(text_list, ncols=0, desc='Encode'):
            input_ids = self.__tokenizer.encode(
                str(text)[:self._max_length - 2],
                add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
                max_length=self._max_length,  # 设定最大文本长度
                padding='max_length',
                # pad_to_max_length=True,  # pad到最大的长度
                return_tensors='pt'  # 返回的类型为pytorch tensor
            )
            input_ids_list.append(input_ids)

        return torch.cat(input_ids_list, dim=0)

    def train(self, train_dataloader, val_dataloader, epochs, device=None, checkpoint=None, **kwargs):
        if device is None:
            device = self.device
        elif device == 'cuda':
            device = torch.device('cuda')
        elif device == 'cpu':
            device = torch.device('cpu')
            if torch.cuda.is_available():
                warnings.warn("cuda is available, you can set (device='cuda') to speed up the program",
                              RuntimeWarning)
        else:
            raise ValueError("device has to be one of (None, 'cpu', 'cuda')")
        args_dict = {
            'checkpoint_save_each_epoch': 0,
            'checkpoint_save_path': './checkpoint'
        }
        for key, value in kwargs.items():
            args_dict[key] = value
        length = len(train_dataloader)
        if checkpoint is not None:
            print('load checkpoint')
            start = self.__load_checkpoint(path=checkpoint)
        else:
            start = 0
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0,
                                                         num_training_steps=length * epochs)
        if (args_dict['checkpoint_save_each_epoch'] > 0) and (not Path(args_dict['save_path']).exists()):
            os.makedirs(args_dict['save_path'])
        for epoch in range(start, epochs):
            self._global_epoch += 1
            info = f'{epoch + 1}/{epochs}'
            self._Train_Loss.reset()
            self._Train_Accuracy.reset()
            self._Val_Loss.reset()
            self._Val_Accuracy.reset()
            self.model.train()
            train_dataloader_bar = tqdm(train_dataloader, ncols=0, desc=f'Epoch {info} Train',
                                        postfix={'Loss': 0, 'Acc': 0})
            for step, (inputs, targets) in enumerate(train_dataloader_bar):
                inputs, targets = inputs.to(device), targets.to(device)
                self.model.zero_grad()
                out = self.model(inputs)[0]
                loss = self.criterion(self._Sigmoid(out), targets)
                out = out.detach().cpu().numpy()
                label_ids = targets.to('cpu').numpy()
                metrics_dict = self.__calculate_metrics(y_true=label_ids, y_pred=out)['samples']
                self._Train_Loss.update(loss.item())
                self._Train_Accuracy.update(metrics_dict['precision'])
                train_dataloader_bar.set_postfix(Loss=self._Train_Loss, Acc=self._Train_Accuracy)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

            self.model.eval()
            val_dataloader_bar = tqdm(val_dataloader, ncols=0, desc=f'Epoch {info}   Val',
                                      postfix={'Loss': 0, 'Acc': 0})
            for inputs, targets in val_dataloader_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.no_grad():
                    out = self.model(inputs)[0]
                    loss = self.criterion(self._Sigmoid(out), targets)
                    out = out.detach().cpu().numpy()
                    label_ids = targets.to('cpu').numpy()
                    metrics_dict = self.__calculate_metrics(y_true=label_ids, y_pred=out)['samples']
                    self._Val_Loss.update(loss.item())
                    self._Val_Accuracy.update(metrics_dict['precision'])
                    val_dataloader_bar.set_postfix(Loss=self._Val_Loss, Acc=self._Val_Accuracy)

            if (args_dict['checkpoint_save_each_epoch'] > 0) and (epoch % args_dict['checkpoint_save_each_epoch'] == 0):
                self.__save_checkpoint(args_dict['save_path'])

    def ROC(self, val_dataloader, show=True, save=False, device=None):
        if device is None:
            device = self.device
        elif device == 'cuda':
            device = torch.device('cuda')
        elif device == 'cpu':
            device = torch.device('cpu')
            if torch.cuda.is_available():
                warnings.warn("cuda is available, you can set (device='cuda') to speed up the program",
                              RuntimeWarning)
        else:
            raise ValueError("device has to be one of (None, 'cpu', 'cuda')")
        outs, labels = [], []
        val_dataloader_bar = tqdm(val_dataloader, desc=f'ROC')
        for inputs, targets in val_dataloader_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                out = self.model(inputs)[0]
                outs.append(out.detach().cpu().numpy())
                labels.append(targets.to('cpu').numpy())
        outs, labels = numpy.vstack(outs), numpy.vstack(labels)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        if os.path.exists(r'./fonts/msyh.ttc'):
            font = FontProperties(fname=r'./fonts/msyh.ttc')
        else:
            warnings.warn("No such file or directory: '.\\Fonts\\msyh.ttc'", RuntimeWarning)
            font = FontProperties()
        colors = ['#6699CC', '#666699', '#99CC66', '#FF9900']
        for i in range(self._num_labels):
            fpr[i], tpr[i], _ = roc_curve(labels[:, i], outs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            if not show:
                print('Class {} ROC area :{:0.2f}'.format(self.labels_name_list[i], roc_auc[i]))
        fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), outs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_fpr = numpy.unique(numpy.concatenate([fpr[i] for i in range(self._num_labels)]))
        mean_tpr = numpy.zeros_like(all_fpr)
        for i in range(self._num_labels):
            mean_tpr += numpy.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= self._num_labels
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        for n, i in enumerate(range(0, self._num_labels, 4)):
            plt.figure(dpi=720)
            plt.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
                     color='#FF6666', linestyle=':', linewidth=2)
            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
                     color='#CCCCCC', linestyle=':', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
            for color, index in zip(colors, range(self._num_labels)[i:i + 4]):
                plt.plot(fpr[index], tpr[index],
                         label='{} ROC curve (area = {:0.2f})'.format(self.labels_name_list[index][:7] + '...' if len(
                             self.labels_name_list[index]) > 7 else self.labels_name_list[index], roc_auc["macro"]),
                         color=color,
                         linewidth=2)
                plt.legend(loc="lower right", fontsize=5, prop=font)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic')
            if save:
                if not Path('./ROC').exists():
                    os.makedirs('./ROC')
                plt.savefig(os.path.join('./ROC', f'ROC_{n}.png'))
            if show:
                plt.show()

    def predict(self, pred_dataloader, threshold=0.5, device=None):
        if device is None:
            device = self.device
        elif device == 'cuda':
            device = torch.device('cuda')
        elif device == 'cpu':
            device = torch.device('cpu')
            if torch.cuda.is_available():
                warnings.warn("cuda is available, you can set (device='cuda') to speed up the program",
                              RuntimeWarning)
        else:
            raise ValueError("device has to be one of (None, 'cpu', 'cuda')")
        self.model.eval()
        preds = []
        for batch, _ in tqdm(pred_dataloader, ncols=0, desc='Predict'):
            with torch.no_grad():
                outputs = self.model(batch.to(device), token_type_ids=None, attention_mask=(batch > 0).to(device))
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                logits = numpy.array(logits > threshold, dtype=int)
                preds.append(logits)
        result = pandas.DataFrame(numpy.vstack(preds))
        result.columns = self.labels_name_list
        return result

    def save_weight(self, path):
        torch.save(self.model.state_dict(), path)

    def load_weight(self, path):
        self.model.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.model, path)


class BERT_Multi_Class_Classification(object):

    def __init__(self, pre_train_path, num_class, max_length=128, lr=2e-5, device=None, from_tf=False,
                 output_attentions=False, output_hidden_states=False):

        self.__tokenizer = BertTokenizer.from_pretrained(pre_train_path, do_lower_case=True)
        self.model = BertForSequenceClassification.from_pretrained(pre_train_path, num_labels=num_class,
                                                                   output_attentions=output_attentions,
                                                                   output_hidden_states=output_hidden_states,
                                                                   from_tf=from_tf)
        self.lr = lr
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self._max_length = max_length
        self._Train_Loss = AverageMeter()
        self._Train_Accuracy = AverageMeter()
        self._Val_Loss = AverageMeter()
        self._Val_Accuracy = AverageMeter()
        self._global_epoch = 0

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
        else:
            self.device = torch.device(device)
            if device == 'cuda':
                self.device = torch.device('cuda')
            elif device == 'cpu':
                self.device = torch.device('cpu')
                if torch.cuda.is_available():
                    warnings.warn("cuda is available, you can set (device='cuda') to speed up the program",
                                  RuntimeWarning)
            else:
                raise ValueError("device has to be one of (None, 'cpu', 'cuda')")
            self.model.to(self.device)

    def __str__(self):
        return str(self.model)

    def __save_checkpoint(self, path):
        file_name = datetime.datetime.now().strftime('checkpoint_%Y_%m_%d_%H_%M_%S.pth.tar')
        torch.save({'epoch': self._global_epoch, 'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, os.path.join(path, file_name))

    def __load_checkpoint(self, path):
        model_checkpoint_dict = torch.load(path)
        self._global_epoch = model_checkpoint_dict['epoch'] - 1
        start = model_checkpoint_dict['epoch']
        self.model.load_state_dict(model_checkpoint_dict['state_dict'])
        self.optimizer.load_state_dict(model_checkpoint_dict['optimizer'])
        return start

    def __flat_accuracy(self, y_true, y_pred):
        pred_flat = numpy.argmax(y_pred, axis=1).flatten()
        labels_flat = y_true.flatten()
        return accuracy_score(labels_flat, pred_flat)

    def encode(self, text_list):
        '''
        用来把文本转换成token并进行pading

        :param text_list:
        :return:
        '''
        input_ids_list = []
        for text in tqdm(text_list, ncols=0, desc='Encode'):
            input_ids = self.__tokenizer.encode(
                str(text)[:self._max_length - 2],
                add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
                max_length=self._max_length,  # 设定最大文本长度
                padding='max_length',
                return_tensors='pt'  # 返回的类型为pytorch tensor
            )
            input_ids_list.append(input_ids)

        return torch.cat(input_ids_list, dim=0)

    def train(self, train_dataloader, val_dataloader, epochs, device=None, checkpoint=None, **kwargs):
        if device is None:
            device = self.device
        elif device == 'cuda':
            device = torch.device('cuda')
        elif device == 'cpu':
            device = torch.device('cpu')
            if torch.cuda.is_available():
                warnings.warn("cuda is available, you can set (device='cuda') to speed up the program",
                              RuntimeWarning)
        else:
            raise ValueError("device has to be one of (None, 'cpu', 'cuda')")
        args_dict = {
            'checkpoint_save_each_epoch': 0,
            'checkpoint_save_path': './checkpoint'
        }
        for key, value in kwargs.items():
            args_dict[key] = value
        length = len(train_dataloader)
        if checkpoint is not None:
            print('load checkpoint')
            start = self.__load_checkpoint(path=checkpoint)
        else:
            start = 0
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0,
                                                         num_training_steps=length * epochs)
        if (args_dict['checkpoint_save_each_epoch'] > 0) and (not Path(args_dict['save_path']).exists()):
            os.makedirs(args_dict['save_path'])
        for epoch in range(start, epochs):
            self._global_epoch += 1
            info = f'{epoch + 1}/{epochs}'
            self._Train_Loss.reset()
            self._Train_Accuracy.reset()
            self._Val_Loss.reset()
            self._Val_Accuracy.reset()
            self.model.train()
            train_dataloader_bar = tqdm(train_dataloader, ncols=0, desc=f'Epoch {info} Train',
                                        postfix={'Loss': 0, 'Acc': 0})
            for step, (inputs, targets) in enumerate(train_dataloader_bar):
                inputs, targets = inputs.to(device), targets.to(device)
                self.model.zero_grad()
                out = self.model(inputs)[0]
                loss = self.criterion(out, targets.squeeze())
                out = out.detach().cpu().numpy()
                label_ids = targets.to('cpu').numpy()
                acc = self.__flat_accuracy(y_true=label_ids, y_pred=out)
                self._Train_Loss.update(loss.item())
                self._Train_Accuracy.update(acc)
                train_dataloader_bar.set_postfix(Loss=self._Train_Loss, Acc=self._Train_Accuracy)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

            self.model.eval()
            val_dataloader_bar = tqdm(val_dataloader, ncols=0, desc=f'Epoch {info}   Val',
                                      postfix={'Loss': 0, 'Acc': 0})
            for inputs, targets in val_dataloader_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.no_grad():
                    out = self.model(inputs)[0]
                    loss = self.criterion(out, targets.squeeze())
                    out = out.detach().cpu().numpy()
                    label_ids = targets.to('cpu').numpy()
                    acc = self.__flat_accuracy(y_true=label_ids, y_pred=out)
                    self._Val_Loss.update(loss.item())
                    self._Val_Accuracy.update(acc)
                    val_dataloader_bar.set_postfix(Loss=self._Val_Loss, Acc=self._Val_Accuracy)

            if (args_dict['checkpoint_save_each_epoch'] > 0) and (epoch % args_dict['checkpoint_save_each_epoch'] == 0):
                self.__save_checkpoint(args_dict['save_path'])

    def predict(self, pred_dataloader, device=None):
        if device is None:
            device = self.device
        elif device == 'cuda':
            device = torch.device('cuda')
        elif device == 'cpu':
            device = torch.device('cpu')
            if torch.cuda.is_available():
                warnings.warn("cuda is available, you can set (device='cuda') to speed up the program",
                              RuntimeWarning)
        else:
            raise ValueError("device has to be one of (None, 'cpu', 'cuda')")
        self.model.eval()
        preds = []
        for batch, _ in tqdm(pred_dataloader, ncols=0, desc='Predict'):
            with torch.no_grad():
                outputs = self.model(batch.to(device), token_type_ids=None, attention_mask=(batch > 0).to(device))
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                logits_flat = numpy.argmax(logits, axis=1).flatten()
                preds.append(logits_flat)
        return pandas.DataFrame({'label': numpy.hstack(preds)})

    def save_weight(self, path):
        torch.save(self.model.state_dict(), path)

    def load_weight(self, path):
        self.model.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.model, path)
