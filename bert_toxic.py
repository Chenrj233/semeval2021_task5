## Imports
import argparse

import pandas as pd
import math
import numpy as np
import random
from seqeval.metrics import classification_report,accuracy_score,f1_score
import torch.nn.functional as F

import torch
import os
from tqdm import tqdm,trange
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig
from transformers import BertForTokenClassification, AdamW
from bertmodel import BertForTokenClassificationCLS, BertForTokenClassificationONATT, BertForTokenClassificationATT, BertForTokenClassificationConcatCLS
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
#%matplotlib inline
from data_parallel import BalancedDataParallel
import shutil

class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["token"].values.tolist(),
                                                           s["label"].values.tolist())]
        self.grouped = self.data.groupby("sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

def get_special_tokens(tokenizer, tag2idx):

    pad_tok = tokenizer.vocab["[PAD]"]
    sep_tok = tokenizer.vocab["[SEP]"]
    cls_tok = tokenizer.vocab["[CLS]"]
    o_lab = tag2idx["O"]

    return pad_tok, sep_tok, cls_tok, o_lab


def flat_accuracy(valid_tags, pred_tags):

    """
    Define a flat accuracy metric to use while training the model.
    """

    return (np.array(valid_tags) == np.array(pred_tags)).mean()





train_file = './data/train.csv'

Classifier = BertForTokenClassificationATT

max_len  = 512
epochs = 8
learning_rate = 5e-6
bert_vocab = 'bert-large-cased'
batch_num = 4
bert_out_address = './models/bert_out_model/bert_att'



if not os.path.exists(bert_out_address):
        os.makedirs(bert_out_address)






df_data = pd.read_csv(train_file ,sep=",",encoding="latin1").fillna(method='ffill')


getter = SentenceGetter(df_data)

sentences = [[s[0] for s in sent] for sent in getter.sentences]



labels = [[s[1] for s in sent] for sent in getter.sentences]

tags_vals = list(set(df_data["label"].values))

tags_vals.append('[PAD]')
tags_vals.append('[CLS]')
tags_vals.append('[SEP]')
tags_vals = set(tags_vals)




tag2idx={
    'I-TOX': 1,
    'O': 0
}



tag2name={tag2idx[key] : key for key in tag2idx.keys()}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu: ", n_gpu)

tokenizer=BertTokenizer.from_pretrained("bert-large-cased", do_lower_case=False)

pad_tok, sep_tok, cls_tok, o_lab = get_special_tokens(tokenizer, tag2idx)



tokenized_texts = []
word_piece_labels = []
i_inc = 0
max_sen = 0
for word_list, label in (zip(sentences, labels)):
    temp_lable = []
    temp_token = []

    # Add [CLS] at the front
    temp_lable.append('O')
    temp_token.append('[CLS]')
    now_len = 1
    is_last = False

    for word, lab in zip(word_list, label):
        token_list = tokenizer.tokenize(word)
        if (is_last):
            break
        for m, token in enumerate(token_list):
            temp_token.append(token)
            if lab == 'B-TOX':
                temp_lable.append('I-TOX')
            elif lab == 'I-TOX':
                temp_lable.append('I-TOX')
            elif lab == 'O':
                temp_lable.append('O')
            now_len = now_len + 1
            if (now_len == max_len - 1):
                is_last = True
                break
                # Add [SEP] at the end
    temp_lable.append('O')
    temp_token.append('[SEP]')
    tokenized_texts.append(temp_token)
    word_piece_labels.append(temp_lable)

    if 5 > i_inc:
        print("No.%d,len:%d" % (i_inc, len(temp_token)))
        print("texts:%s" % (" ".join(temp_token)))
        print("No.%d,len:%d" % (i_inc, len(temp_lable)))
        print("lables:%s" % (" ".join(temp_lable)))
    i_inc += 1




input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=max_len, dtype="long", truncating="post", padding="post",
                          value=pad_tok)


tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in word_piece_labels],
                     maxlen=max_len, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")


attention_masks = [[int(i != pad_tok) for i in ii] for ii in input_ids]



tr_inputs, val_inputs, tr_tags, val_tags,tr_masks, val_masks = train_test_split(input_ids, tags,attention_masks,
                                                            random_state=7, test_size=0.11)

tr_inputs = torch.LongTensor(tr_inputs)
val_inputs = torch.LongTensor(val_inputs)
tr_tags = torch.LongTensor(tr_tags)
val_tags = torch.LongTensor(val_tags)
tr_masks = torch.LongTensor(tr_masks)
val_masks = torch.LongTensor(val_masks)




model = Classifier.from_pretrained(bert_vocab,num_labels=len(tag2idx))

model.cuda()


train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
# Drop last can make batch training better for the last one
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_num,drop_last=True)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_num)




if n_gpu >1:

    model = BalancedDataParallel(2, model,dim=0)


max_grad_norm = 1.0
num_train_optimization_steps = int( math.ceil(len(tr_inputs) / batch_num) / 1) * epochs

FULL_FINETUNING = True
if FULL_FINETUNING:
    # Fine tune model all layer parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0},
    ]

else:
    # Only fine tune classifier parameters
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

total_steps = len(train_dataloader) * epochs
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-9)#5e-3
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)




seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
training_states = []
dev_best_acc = 0.0
dev_best_f1 = 0.1


tokenizer.save_vocabulary(bert_out_address)


print("***** Running training *****")
print("  Num examples = %d" % (len(tr_inputs)))
print("  Batch size = %d" % (batch_num))
print("  Num steps = %d" % (num_train_optimization_steps))
epoch = 0
tolstep = 0
stop_count = 0
min_loss = 1e9
for _ in trange(epochs, desc="Epoch"):
    print()
    epoch += 1
    tr_loss = 0
    tr_accuracy = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    model.train()
    for step, batch in zip(tqdm(range(len(train_dataloader)),position=0,desc="Data"), train_dataloader):
        # add batch to gpu
        tolstep += 1
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        optimizer.zero_grad()
        # forward pass
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        loss, tr_logits = outputs[:2]

        if n_gpu > 1:
            # When multi gpu, average it
            loss = loss.mean()




        # backward pass
        loss.backward()
        optimizer.step()

        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # Subset out unwanted predictions on CLS/PAD/SEP tokens
        preds_mask = (
                (b_input_ids != cls_tok)
                & (b_input_ids != pad_tok)
                & (b_input_ids != sep_tok)
        )

        tr_label_ids = torch.masked_select(b_labels, (preds_mask == 1))

        tr_batch_preds = torch.argmax(tr_logits[preds_mask], axis=1)
        tr_batch_preds = tr_batch_preds.detach().cpu().numpy()
        tr_batch_labels = tr_label_ids.to("cpu").numpy()
        tr_preds.extend(tr_batch_preds)
        tr_labels.extend(tr_batch_labels)

        # Compute training accuracyG:
        tmp_tr_accuracy = flat_accuracy(tr_batch_labels, tr_batch_preds)
        tr_accuracy += tmp_tr_accuracy


        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)

        # update parameters

        scheduler.step()

    tr_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps

    # Print training loss and accuracy per epoch
    print()
    print(f"Train loss: {tr_loss}")
    print(f"Train accuracy: {tr_accuracy}")
    print()
    print("Running Validation...")
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions, true_labels = [], []

    for step, batch in enumerate(valid_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )
            tmp_eval_loss, logits = outputs[:2]

        # Subset out unwanted predictions on CLS/PAD/SEP tokens
        preds_mask = (
                (b_input_ids != cls_tok)
                & (b_input_ids != pad_tok)
                & (b_input_ids != sep_tok)
        )

        label_ids = torch.masked_select(b_labels, (preds_mask == 1))

        val_batch_preds = torch.argmax(logits[preds_mask], axis=1)
        val_batch_preds = val_batch_preds.detach().cpu().numpy()
        val_batch_labels = label_ids.to("cpu").numpy()
        predictions.extend(val_batch_preds)
        true_labels.extend(val_batch_labels)

        tmp_eval_accuracy = flat_accuracy(val_batch_labels, val_batch_preds)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

    # Evaluate loss, acc, conf. matrix, and class. report on devset
    pred_tags = [[tag2name[i] for i in predictions]]
    valid_tags = [[tag2name[i] for i in true_labels]]
    cl_report = classification_report(valid_tags, pred_tags)
    eval_loss = eval_loss / nb_eval_steps
    tmp_accuracy = accuracy_score(valid_tags, pred_tags) 
    if tmp_accuracy > dev_best_acc:
        dev_best_acc = tmp_accuracy
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        output_model_file = os.path.join(bert_out_address, "pytorch_model.bin")
        output_config_file = os.path.join(bert_out_address, "config.json")

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    # Report metrics
    f1 = f1_score(valid_tags, pred_tags)
    if f1 > dev_best_f1:
      dev_best_f1 = f1
      model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

      output_model_file = os.path.join(bert_out_address, "f1_pytorch_model.bin")
      output_config_file = os.path.join(bert_out_address, "f1_config.json")

      torch.save(model_to_save.state_dict(), output_model_file)
      model_to_save.config.to_json_file(output_config_file)


    print(f"Classification Report:\n {cl_report}")
    print(f"Validation loss: {eval_loss}")
    print("f1 socre: %f" % f1)
    print("Accuracy score: %f" % (accuracy_score(valid_tags, pred_tags)))

    training_states.append(
        {
            'epoch': epoch,
            'Trainning Loss': tr_loss,
            'Trainning Accur': tr_accuracy,
            'Valid Loss': eval_loss,
            'Valid Accur': tmp_accuracy,
            'Valid f1': f1
        }
    )
    df_states = pd.DataFrame(data=training_states)
    df_states = df_states.set_index('epoch')
    df_states.to_csv(os.path.join(bert_out_address, "training_epoch_states.csv"))

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    output_model_file = os.path.join(bert_out_address, "last_pytorch_model.bin")
    output_config_file = os.path.join(bert_out_address, "last_config.json")
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)


print('Training complete!')





pd.set_option('precision', 3)

df_states = pd.DataFrame(data=training_states)
df_states = df_states.set_index('epoch')
df_states.to_csv(os.path.join(bert_out_address, "training_epoch_states.csv"))
print(df_states)




