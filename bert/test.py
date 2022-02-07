from model import BertForSequenceClassification
from main import get_predictions
from dataloader import getTestLoader
from sklearn.metrics import f1_score
from sklearn import metrics
import torch
import numpy as np


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    PRETRAINED_MODEL_NAME = "bert-base-multilingual-cased"
    NUM_LABELS = 2

    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

    testloader = getTestLoader()

    model.load_state_dict(torch.load("model0_bert", map_location = device))

    model.to(device)


    for dataloader in testloader:

        predictions, _ = get_predictions(model, dataloader, compute_acc=False)


    predictions = predictions.cpu().numpy()

    # print(val_y.shape)
    # print(predictions.shape)

    # print(f1_score(val_y, predictions, average='macro'))
    # print(metrics.confusion_matrix(val_y, predictions))
    # print(metrics.classification_report(val_y, predictions, digits=3))

    np.savetxt('bert_feature_0.csv', predictions, delimiter=',', fmt='%i')

if __name__ == "__main__": main()