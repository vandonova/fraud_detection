from flask import Flask, request, render_template
from predict import get_X
from pymongo import MongoClient
import cPickle as pickle
import matplotlib.pyplot as plt
import os
import pandas as pd
import requests
import socket
import time

app = Flask(__name__)
REGISTER_URL = 'http://10.3.35.95:5000/register'

with open('model/LogisticRegressionmodel.pkl') as f:
    lr = pickle.load(f)

client = MongoClient()
db = client['fraud_db']


def register_for_ping(ip, port):
    '''Register to receive incoming data points
    Args:
        ip: IP address
        port: port number (should be stored in .env file)
    Returns:
        None
    '''

    registration_data = {'ip': ip, 'port': port}
    requests.post(REGISTER_URL, data=registration_data)


def risk(prob):
    '''Takes in the probability of fraud and returns a risk assessment coupled with a color.
    Args:
        prob (float): probability of fraud
    Returns:
        (color, risk): risk category based on probability thresholds
    '''

    if prob > 0.5:
        return ("red", "High Risk")
    elif prob > 0.1:
        return ("orange", "Medium Risk")
    else:
        return ("blue", "Low Risk")


def plot_features(lr_classifier, X, save=True):
    '''Uses the coefficients of the Logistic Regression model to plot how each feature confributed to the prediction for an event in X
    Args:
        lg_classifier: fitted Logistic Regression classifier
        X: data, each row being a different event
        save: whether to save the figure or not
    Returns:
        None
    '''

    plt.figure()
    pred_ = (X.values * 2 - 1) * lr_classifier.coef_.reshape(11,)

    pred_df = pd.DataFrame({"pred": pred_})
    pred_df['positive'] = pred_df['pred'] > 0
    pred_df['pred'].plot(
        kind='barh', color=pred_df.positive.map({True: 'r', False: 'b'}))

    plt.xlim((-4, 4))
    plt.title("Feature Influence")
    plt.xlabel("Magnitude")
    plt.yticks(range(len(X)), ['Max price below $80', 'Has free tickets',
                               'Has analytics', 'Has a logo',
                               'Has a payout type',
                               'Has a previous payout',
                               'Has a sketchy name', 'Has a sketchy email',
                               'Has a twitter', 'Has a facebook',
                               'Less than one ticket sold'])
    plt.tight_layout()
    if save:
        plt.savefig('static/images/feature_importances.png')


@app.route('/score', methods=['POST'])
def score():
    '''Asynchronously receives post methods from server, stores in mongodb
    '''

    json_line = request.json
    time_stamp = time.time()
    json_line['timestamp'] = time_stamp
    db.pages.insert_one(json_line)
    return ""


@app.route('/')
def index():
    '''Renders the landing page template
    '''

    return render_template('starter_template.html')


@app.route('/prediction')
def predict_one():
    '''If there are events stored in the database, take the most recent one and predict whether it is fraud of not
    Args:
        None
    Returns:
        Renders template with risk score and features plot
    '''

    if db.pages.find():
        cursor = db.pages.find()
        df = pd.DataFrame(list(cursor))
        X = get_X(df)
        X = X.iloc[-1, :]
        y = lr.predict_proba(X)[:, 1]

        prob = y[0]  # probability of risk for most recent event
        plot_features(lr, X, save=True)
        return render_template('prediction_template.html', risk=risk(prob)[1],
                               color=risk(prob)[0])
    else:
        return render_template('prediction_template_no_image.html',
                               risk="No Data", color="black")


@app.route('/summary')
def summary():
    '''Renders a template with total number of events and total number of frauds
    '''

    if db.pages.find():
        cursor = db.pages.find()
        df = pd.DataFrame(list(cursor))
        X = get_X(df)
        y = lr.predict(X) * 1

        total = len(y)
        frauds = sum(y)
        return render_template('overall_template.html', total=str(total),
                               frauds=frauds)
    else:
        return render_template('prediction_template_no_image.html',
                               risk="No Data", color="black")


@app.route('/model_building')
def model_building():
    '''EDA overview template
    '''
    return render_template('plot_wanru.html')


if __name__ == '__main__':
    # Load port number
    if os.path.exists('.env'):
        for line in open('.env'):
            var = line.strip().split('=')
            if len(var) == 2:
                os.environ[var[0]] = var[1]
                sys.stdout.flush()

    # Register for pinging service
    ip_address = socket.gethostbyname(socket.gethostname())
    ip_address = '10.3.34.52'
    print "Attempting to register %s:%d" % (ip_address, PORT)
    register_for_ping(ip_address, str(PORT))

    # Start Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)
