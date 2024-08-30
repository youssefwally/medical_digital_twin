from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

def draw_plot(datadir, DS, embeddings, fname, max_nodes=None):
    return
    import seaborn as sns
    graphs = read_graphfile(datadir, DS, max_nodes=max_nodes)
    labels = [graph.graph['label'] for graph in graphs]

    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)
    print('fitting TSNE ...')
    x = TSNE(n_components=2).fit_transform(x)

    plt.close()
    df = pd.DataFrame(columns=['x0', 'x1', 'Y'])

    df['x0'], df['x1'], df['Y'] = x[:,0], x[:,1], y
    sns.pairplot(x_vars=['x0'], y_vars=['x1'], data=df, hue="Y", size=5)
    plt.legend()
    plt.savefig(fname)

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

def logistic_classify(x, y):
    nb_classes = np.unique(y).shape[0]
    xent = nn.CrossEntropyLoss()
    hid_units = x.shape[1]

    accs = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    for train_index, test_index in kf.split(x, y):
        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls= y[train_index], y[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs).cuda(), torch.from_numpy(train_lbls).cuda()
        test_embs, test_lbls= torch.from_numpy(test_embs).cuda(), torch.from_numpy(test_lbls).cuda()


        log = LogReg(hid_units, nb_classes)
        log.cuda()
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        best_val = 0
        test_acc = None
        for it in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc.item())
    return np.mean(accs)

def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies)

def randomforest_classify(x, y, search):
    kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):
        print("Split Done", flush=True)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            print("starting Search", flush=True)
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=3, scoring='accuracy', verbose=0)
        else:
            classifier = RandomForestClassifier()
        
        print("starting fit", flush=True)
        classifier.fit(x_train, y_train)
        print("Fit Done", flush=True)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    ret = np.mean(accuracies)
    return ret

def randomforest_regressor(x, y, search):
    kf = KFold(n_splits=2, shuffle=True, random_state=None)
    r2_scores = []
    for train_index, test_index in kf.split(x, y):
        print("Split Done", flush=True)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            print("starting Search", flush=True)
            regressor = GridSearchCV(RandomForestRegressor(), params, cv=3, scoring='r2', verbose=0)
        else:
            regressor = RandomForestRegressor()

        print("starting fit", flush=True)
        regressor.fit(x_train, y_train)
        print("Fit Done", flush=True)
        r2_scores.append(r2_score(y_test, regressor.predict(x_test)))
    ret = np.mean(r2_scores)
    return ret

def linearsvc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies)

def evaluate_embedding(embeddings, labels, classification=True, search=True):
    print(classification, flush=True)
    if(classification):
        labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)
    # print(x.shape, y.shape)

    if(classification):
        logreg_accuracies = [logistic_classify(x, y) for _ in range(1)]
        # print(logreg_accuracies)
        print('LogReg', np.mean(logreg_accuracies))
    else:
        logreg_accuracies = [0]

    # svc_accuracies = [svc_classify(x,y, search) for _ in range(1)]
    # # print(svc_accuracies)
    # print('svc', np.mean(svc_accuracies))

    # linearsvc_accuracies = [linearsvc_classify(x, y, search) for _ in range(1)]
    # # print(linearsvc_accuracies)
    # print('LinearSvc', np.mean(linearsvc_accuracies))

    if(classification):
        randomforest_accuracies = [randomforest_classify(x, y, search) for _ in range(1)]
        # print(randomforest_accuracies)
    else:
        randomforest_accuracies = [randomforest_regressor(x, y, search) for _ in range(1)]
    print('randomforest', np.mean(randomforest_accuracies))

    # return np.mean(logreg_accuracies), np.mean(svc_accuracies), np.mean(linearsvc_accuracies), np.mean(randomforest_accuracies)
    return np.mean(logreg_accuracies), 0, 0, np.mean(randomforest_accuracies)

if __name__ == '__main__':
    evaluate_embedding('./data', 'ENZYMES', np.load('tmp/emb.npy'))
