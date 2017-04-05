from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
from predict import get_X
import pandas as pd


class MyModels(object):

    def __init__(self):
        self.nb = MultinomialNB(alpha=.5)
        self.logreg = LogisticRegression()
        self.rf = RandomForestClassifier(oob_score=True)
        self.models = [self.nb, self.logreg, self.rf]


def standard_confusion_matrix(y_true, y_pred):
    '''Converts the format of the sklearn confusion matrix
    Args:
        y_true, y_pred
    Returns:
        confusion matrix: [[tp, fp], [fn, tn]]
    '''

    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])


def profit_curve(cb, predict_probas, y_true):
    '''Calculates the profit for each model
    Args:
        cb: cost benefit matrix formatted as [[tp, fp], [fn, tn]]
        predict_probas: predicted probabilities for positive class
        labels: actual y labels
    Returns:
        profits (list): list of profits for each threshold
    '''

    # Cost benefit is formatted, actual as x, predicted as y
    # invert to match confusion matrix output
    profits = []
    for T in np.linspace(0, 1, 101):
        preds = (predict_probas > T) * 1
        # tp = sum(labels & preds)
        # tn = sum((not labels) & (not preds))
        # fp = sum(preds) - tp
        # fn = sum(labels) - tp
        cm = standard_confusion_matrix(y_true, preds)
        profit = sum(sum(cm * cb)) / len(y_true)
        profits.append(profit)

    return profits


def plot_profit_curve(model, label, costbenefit, X_train, X_test, y_train, y_test):
    '''Plots the profit curve for the classifier'''
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    profits = profit_curve(costbenefit, preds, y_test)
    plt.plot(np.linspace(0, 1, 101), profits, label=label)


def get_data(filename):
    '''Reads in, prepares, and splits the data'''
    df = pd.read_json(filename)
    df['fraud'] = df['acct_type'].apply(lambda x: 'fraud' in x)
    X = get_X(df)
    y = df['fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data('data/data.json')
    models = MyModels()
    profit_mat = np.array([[0, -50], [-1000, 0]])
    for model in models.models:
        model.fit(X_train, y_train)
        print 'Training accuracy: ', model.score(X_train, y_train)
        if model.__class__.__name__ == 'RandomForestClassifier':
            print 'OOB score: ', model.oob_score_
        print 'Test accuracy: ', model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        print 'Test F1 score: ', f1_score(y_test, y_pred)
        print confusion_matrix(y_test, y_pred)
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=model.__class__.__name__)
        # plt.show()
        with open('models/' + model.__class__.__name__ + 'Model.pkl', 'w') as f:
            pickle.dump(model, f)

    plt.title("ROC Plots")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.show()

    # for model in models.models:
    #     plot_profit_curve(model, model.__class__.__name__, profit_mat, X_train, X_test, y_train, y_test)
    #     plt.title("Profit Curves")
    #     plt.xlabel("Threshold")
    #     plt.ylabel("Profit")
    #     plt.legend(loc='lower right')
    #     plt.show()
