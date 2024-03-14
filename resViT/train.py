import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sn
import numpy as np
# import random
import torch
# import torch.nn as nn
# import math
from torchmetrics import ConfusionMatrix
from sklearn.metrics import f1_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, accuracy_score
import time
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
import matlab.engine
pr2np = matlab.engine.start_matlab()

def getFeatures(dataLoader):
    features = []
    labels = []
    for images, class_labels in dataLoader:
        images = images.numpy()
        features.extend(images)
        labels.extend(class_labels.numpy())

    features = [feat.flatten() for feat in features]
    features = np.array(features)
    labels = np.array(labels)
    return features, labels

def validate_svm(model, ValLoader, numClass, task = 'multiclass'):
    start_time = time.time()
    x_test, tr_labels = getFeatures(ValLoader)
    pred = model.predict(x_test)
    end_time = time.time()
    pred = pred.reshape(-1)
    prec = precision_score(tr_labels, pred, average='macro')
    f1 = f1_score(tr_labels, pred, average='macro')
    rec = recall_score(tr_labels, pred, average='macro')
    cm = confusion_matrix(tr_labels, pred)
    return accuracy_score(tr_labels, pred), cm, prec, f1, rec, end_time-start_time, tr_labels, pred

def validate_pr(model, ValLoader, numClass, task = 'multiclass'):
    # cm_metric = ConfusionMatrix(task = task, num_classes=numClass).to(device)
    correct = 0
    total = 0
    tr_labels = np.array([])
    pred = np.array([])
    model.eval()
    with torch.no_grad():
        for data in ValLoader:
            inputs, labels = data[0].to(device), data[1].to(device)
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # cm_metric.update(predicted, labels)
            tr_labels = np.concatenate([tr_labels, labels.cpu().data.numpy()], axis=0)
            pred = np.concatenate([pred, predicted.cpu().data.numpy()], axis=0)
        # cm = cm_metric.compute().detach().cpu().numpy()
    pred = np.array(pr2np.intp_pr(tr_labels, pred))
    # pred = np.array(pr2np.intp_prN(tr_labels, pred))
    pred = pred.reshape(-1)
    prec = precision_score(tr_labels, pred, average='macro')
    f1 = f1_score(tr_labels, pred, average='macro')
    rec = recall_score(tr_labels, pred, average='macro')
    cm = confusion_matrix(tr_labels, pred)
    return accuracy_score(tr_labels, pred), cm, prec, f1, rec, end_time-start_time, tr_labels, pred


def validate(model, ValLoader, numClass, task = 'multiclass'):
    # cm_metric = ConfusionMatrix(task = task, num_classes=numClass).to(device)
    correct = 0
    total = 0
    tr_labels = np.array([])
    pred = np.array([])
    model.eval()
    with torch.no_grad():
        for data in ValLoader:
            inputs, labels = data[0].to(device), data[1].to(device)
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # cm_metric.update(predicted, labels)
            tr_labels = np.concatenate([tr_labels, labels.cpu().data.numpy()], axis=0)
            pred = np.concatenate([pred, predicted.cpu().data.numpy()], axis=0)
        # cm = cm_metric.compute().detach().cpu().numpy()
    prec = precision_score(tr_labels, pred, average='macro')
    f1 = f1_score(tr_labels, pred, average='macro')
    rec = recall_score(tr_labels, pred, average='macro')
    cm = confusion_matrix(tr_labels, pred)
    return (correct / total) * 100, cm, prec, f1, rec, end_time-start_time, tr_labels, pred


def disp_conf_matrix(cm, class_names = ['N', 'IR', 'B', 'OR']):
    fig, ax = plt.subplots(figsize=(4,2))
    ConfusionMatrixDisplay(cm, display_labels = class_names).plot(ax=ax)
    


# Main Training loop

def train_pr(Model, TrainLoader, ValLoader, optC, criterion, tr_loss = [], val_loss = [], num_epochs=20):
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        correct_tr = 0
        total_tr = 0
        labels_tr = np.array([])
        pred_tr = np.array([])
        correct_val = 0
        total_val = 0
        labels_val = np.array([])
        pred_val = np.array([])
        for i, data in enumerate(TrainLoader, 0):
            #   Model.train()
              inputs, labels = data[0].to(device), data[1].to(device)
    
              torch.autograd.set_detect_anomaly(True)
              optC.zero_grad()
              predictions = Model(inputs)
              ClassifierLoss = criterion(predictions, labels)
              ClassifierLoss.backward(retain_graph=True)
              optC.step()
            
        #       # Evaluate model
              Model.eval()
              with torch.no_grad():
                  outputs = Model(inputs)
                  _, predicted = torch.max(outputs.data, 1)
                  total_tr += labels.size(0)
                  correct_tr += (predicted == labels).sum().item()
                  labels_tr = np.concatenate([labels_tr, labels.cpu().data.numpy()], axis=0)
                  pred_tr = np.concatenate([pred_tr, predicted.cpu().data.numpy()], axis=0)
        train_acc = pr2np.CompInfN(epoch+1, labels_tr, pred_tr, 0)#(correct_tr / total_tr) * 100
        for data in ValLoader:
              with torch.no_grad():
                  inputs, labels = data[0].to(device), data[1].to(device)
                  outputs = Model(inputs)
                  _, predicted = torch.max(outputs.data, 1)
                  total_val += labels.size(0)
                  correct_val += (predicted == labels).sum().item()
                  labels_val = np.concatenate([labels_val, labels.cpu().data.numpy()], axis=0)
                  pred_val = np.concatenate([pred_val, predicted.cpu().data.numpy()], axis=0)
        val_acc = pr2np.CompInfN(epoch+1, labels_val, pred_val, 1) #(correct_val / total_val) * 100
        print('[%d/%d]   Train Accuracy = %.2f    Val Accuracy = %.2f'  %(epoch, num_epochs, train_acc, val_acc))
    return Model

def train(Model, TrainLoader, ValLoader, optC, criterion, tr_loss = [], val_loss = [], num_epochs=20):
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        correct_tr = 0
        total_tr = 0
        labels_tr = np.array([])
        pred_tr = np.array([])
        correct_val = 0
        total_val = 0
        trLoss = []
        valLoss = []
        labels_val = np.array([])
        pred_val = np.array([])
        for i, data in enumerate(TrainLoader, 0):
              Model.train()
              inputs, labels = data[0].to(device), data[1].to(device)
    
              torch.autograd.set_detect_anomaly(True)
              optC.zero_grad()
              predictions = Model(inputs)
              ClassifierLoss = criterion(predictions, labels)
              ClassifierLoss.backward(retain_graph=True)
              optC.step()
            
              # Evaluate model
              Model.eval()
              with torch.no_grad():
                  outputs = Model(inputs)
                  trLoss.append(criterion(outputs, labels))
                  _, predicted = torch.max(outputs.data, 1)
                  total_tr += labels.size(0)
                  correct_tr += (predicted == labels).sum().item()
                  labels_tr = np.concatenate([labels_tr, labels.cpu().data.numpy()], axis=0)
                  pred_tr = np.concatenate([pred_tr, predicted.cpu().data.numpy()], axis=0)
        train_acc = (correct_tr / total_tr) * 100
        # tr_loss.append(sum(trLoss) / len(trLoss))
        for data in ValLoader:
              with torch.no_grad():
                  inputs, labels = data[0].to(device), data[1].to(device)
                  outputs = Model(inputs)
                  valLoss.append(criterion(outputs, labels))
                  _, predicted = torch.max(outputs.data, 1)
                  total_val += labels.size(0)
                  correct_val += (predicted == labels).sum().item()
                  labels_val = np.concatenate([labels_val, labels.cpu().data.numpy()], axis=0)
                  pred_val = np.concatenate([pred_val, predicted.cpu().data.numpy()], axis=0)
        val_acc = (correct_val / total_val) * 100
          # val_loss.append(sum(valLoss) / len(valLoss))
        print('[%d/%d]   Train Accuracy = %.2f    Val Accuracy = %.2f'  %(epoch, num_epochs, train_acc, val_acc))
    return Model