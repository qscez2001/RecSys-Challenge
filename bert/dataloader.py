from preprocess import preprocess, get_test
# choose(0,1,2,3 model)
# train_x, val_x, train_y, val_y = preprocess(0)
test_x = get_test()
# print(len(train_x), len(val_x))

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

PRETRAINED_MODEL_NAME = "bert-base-multilingual-cased" 

# tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

class TweetsDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, list_IDs, labels=None):
        self.list_IDs = list_IDs
        self.labels = labels
        # self.tokenizer = tokenizer 

    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        X = self.list_IDs[idx]
        # feature = self.list_IDs[idx][1]
        # y = self.labels[idx]    
        if self.labels:
            label_id = self.labels[idx]
            label_tensor = torch.tensor(label_id)
        else:
            label_id = 0
            label_tensor = torch.tensor(label_id)
        # word_pieces = ["[CLS]"]
        # tokens = self.tokenizer.tokenize(X)
        # word_pieces += X + ["[SEP]"] 
        len_a = len(X)  
        # print(word_pieces)

        # 第二個句子的 BERT tokens
        # tokens_b = self.tokenizer.tokenize(feature)
        # word_pieces += tokens_b + ["[SEP]"]
        # len_b = len(word_pieces) - len_a   
        
        # 將整個 token 序列轉換成索引序列
        # ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(X)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        # segments_tensor = torch.tensor([0] * len_a + [1] * len_b, dtype=torch.long)
        segments_tensor = torch.tensor([0] * len_a, dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return len(self.list_IDs)

def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    

    label_ids = torch.stack([s[2] for s in samples])
    
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, 
                                    batch_first=True)
    
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids


def create_dataloader(idx):
    train_x, val_x, train_y, val_y = preprocess(idx)
    
    trainset = TweetsDataset(train_x, train_y)
    valset = TweetsDataset(val_x, val_y)
    
    BATCH_SIZE = 8
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch, shuffle = True)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)

    return trainloader, valloader


def getTestLoader():


    testloader = []

    testset = TweetsDataset(test_x)
    # testloader = DataLoader(testset, batch_size=1, collate_fn=create_mini_batch)
    testloader.append(DataLoader(testset, batch_size=1, collate_fn=create_mini_batch))

    return testloader

def getAllLoader():

    all_loader = []
    X = np.concatenate((train_x, val_x))
    y = np.concatenate((train_y, val_y))

    dataset = TweetsDataset(X, y)

    all_loader.append(DataLoader(dataset, batch_size=1, collate_fn=create_mini_batch))

    return all_loader

