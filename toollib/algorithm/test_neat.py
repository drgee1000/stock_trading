"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
from zipline.api import (order, record, symbol, order_target_percent, set_benchmark, set_long_only, schedule_function,
                         date_rules, time_rules)
from zipline import run_algorithm
import os.path
import math
import numpy as np
# Pandas library: https://pandas.pydata.org/
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from toollib.Data.TA.TA_indicator import TA
from toollib.Neat.evolve import Neat

MODEL_NAME = ''
SYMBOL = ''
stocks_bundle = 'custom-stocks-csvdir-bundle'
currency_bundle = 'custom-currency-csvdir-bundle'

def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    feature_num = 11
    context.model_created = False
    context.p = None
    context.orders_submitted = False
    large_num = 9999999
    least_num = 0
    context.n_components = feature_num
    context.security = symbol(SYMBOL)  # Trade SPY
    set_benchmark(symbol(SYMBOL))  # Set benchmarks
    fitness_index = 0
    intervals = 20
    arg1 = 0.7
    arg2 = 0.8
    context.model = Neat(fitness_index,intervals,arg1,arg2)
    context.model2 = SVC(kernel='rbf', tol=1e-3, random_state=0, gamma=0.2, C=10.0, verbose=True)  # 8.05 for SVM model
    context.model3 = KNeighborsClassifier(n_neighbors=feature_num, p=3, metric='minkowski')  # 7.05 for  model
    context.model5 = DecisionTreeClassifier(criterion='entropy', max_depth=feature_num, random_state=0)
    context.model4 = RandomForestClassifier(criterion='entropy', n_estimators=feature_num, random_state=1,
                                            n_jobs=2)  # 5.2 for randomforest
    context.model1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    context.modellist = {'SVM':context.model2,'KNeighbors':context.model3,'DecisionTree':context.model,'RandomForest':context.model4,'LogisticRegression':context.model1}
    context.lookback = 350  # Look back 62 days
    context.history_range = 350  # Only consider the past 400 days' history
    context.threshold = 4.05
    context.longprices = large_num
    context.shortprices = least_num
    set_long_only()
    # Generate a new model every week
    schedule_function(create_model, date_rules.week_end(), time_rules.market_close(minutes=10))
    """
    # Generate a new model every week
    schedule_function(create_model1, date_rules.week_end(), time_rules.market_close(minutes=10))
    """

    # Trade at the start of every day
    #schedule_function(rebalance, date_rules.every_day(), time_rules.market_open(minutes=1))

def handle_data(context, data):
    pass
def create_model(context, data):
    # Get the relevant daily prices
    model = context.model
    recent_prices = data.history(context.security, 'price', context.history_range, '1d').values
    recent_volume = data.history(context.security, 'volume', context.history_range, '1d').values
    recent_high = data.history(context.security, 'high', context.history_range, '1d').values
    recent_low = data.history(context.security, 'low', context.history_range, '1d').values
    recent_dates = data.history(context.security, 'price', context.lookback + 1, '1d').index
    #print(recent_dates)
    input_, target_ = getTrainingWindow(recent_high,recent_low,recent_prices, recent_volume,recent_dates)
    y = np.delete(target_, 0, 1)
    y = np.ravel(y)
    X_normalized_ = preprocessing.normalize(input_, norm='l2')
    sc = preprocessing.MinMaxScaler()
    sc.fit(X_normalized_)
    X_std = sc.transform(X_normalized_)
    # feature selection to input features (context.n_components)
    X_new = SelectKBest(chi2, k=context.n_components).fit_transform(X_std, y)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=0)
    print(X_train.shape)
    config_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'Neat/config'))
    if not(context.model_created):
        context.p = model.run(config_path, X_train, X_test, y_train, y_test)
        context.model_created = True
    else:
        context.p = model.run_online_population(config_path,X_train, X_test, y_train, y_test,context.p)
   # model.fit(X_train, y_train)


def getSector(length, sector):
    date = 1
    Sector_ = []
    for i in range(length):
        Sector_.append([date, sector])
        date = date + 1
    return Sector_

def mergeMatrice(Matrix_A, Matrix_B):
    return np.concatenate((np.delete(Matrix_A, 0, 1), np.delete(Matrix_B, 0, 1)), axis=1)


def getTarget(price, threshold, horizon):
    date = 1
    dataset_prices = []
    price_prev = 0
    labeled_target = []
    for data in price:
        # print(data)
        if (math.isnan(data)):
            continue
        # price change: one day ratio
        dataset_prices.append(data)

    for data_price in dataset_prices:
        if (price_prev == 0):
            price_prev = data_price
            price_delta = 0
        else:
            price_next = data_price
            price_delta = price_next - price_prev
            price_prev = price_next

        if not ((date + horizon) > np.size(dataset_prices)):
            if (price_delta > 0):
                if ((dataset_prices[date - 1 + horizon] / dataset_prices[date - 1]) >= (threshold + 1)):
                    target = 1  # 1 means buy

                elif ((dataset_prices[date - 1 + horizon] / dataset_prices[date - 1]) <= (1 - threshold)):
                    target = -1  # -1 means sell

                else:
                    target = 0  # 0 means hold

            else:
                if ((dataset_prices[date - 1] / dataset_prices[date - 1 + horizon]) >= (1 + threshold)):
                    target = 1  # 1 means buy

                elif ((dataset_prices[date - 1] / dataset_prices[date - 1 + horizon]) <= (1 - threshold)):
                    target = -1  # -1 means sell

                else:
                    target = 0  # 0 means sell
        else:
            target = np.nan
        labeled_target.append([date, target])
        date = date + 1
    return labeled_target


def getTrainingWindow(high, low, prices, volume,dates):
    # Query historical pricing data
    date = 1
    ta_ = TA(dates)
    ta_.addFeature(['PM', 'EMA', 'OBV', 'MA', 'MACD', 'STOCH', 'CCI', 'AD'], dates, [prices, volume, high, low])
    input_data_set = ta_.getInputMatrix()
    tar = getTarget(prices, 0.015, 4)
    for data in input_data_set:
        if (any(np.isnan(ele) for ele in data) or np.isnan(tar[date - 1]).any()):
            input_data_set = np.delete(input_data_set, date - 1, 0)
            tar = np.delete(tar, date - 1, 0)
        else:
            date = date + 1

    return input_data_set, tar


def rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """
    # Get recent prices
    model = context.model
    recent_prices = data.history(context.security, 'price', context.history_range, '1d').values
    recent_volume = data.history(context.security, 'volume', context.history_range, '1d').values
    recent_high = data.history(context.security, 'high', context.history_range, '1d').values
    recent_low = data.history(context.security, 'low', context.history_range, '1d').values
    recent_dates = data.history(context.security, 'price', context.lookback + 1, '1d').index
    input_, _ = getTrainingWindow(recent_high, recent_low, recent_prices, recent_volume,recent_dates)
    y = np.delete(_, 0, 1)
    y = np.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(input_, y, test_size=0.3, random_state=0)
    X_normalized_ = preprocessing.normalize(X_test, norm='l2')
    sc = preprocessing.MinMaxScaler()
    sc.fit(X_normalized_)
    X_std = sc.transform(X_normalized_)
    # feature selection to input features (context.n_components)
    X_new = SelectKBest(chi2, k=context.n_components).fit_transform(X_std, y_test)
    X_test = X_new[-1, :]

    for stock in context.portfolio.positions:
        print(context.portfolio.positions[stock].amount)
    if not context.orders_submitted:
        order(context.security, 10000)
        print('Initial orders submitted')
        context.orders_submitted = True
    try:
        if model:  # Check if our model is generated
            # Predict using our model and the recent prices
            X_test_ = X_test.reshape(1, -1)
            prediction = model.predict(X_test_)
            prediction_accuracy = model.predict(X_new[-10:, :])  # predict in past 10 days
            accuracy = accuracy_score(np.ravel(np.delete(_, 0, 1))[-10:], np.array(prediction_accuracy).round())
            print('Accuracy: %.2f' % accuracy)
            record(accuracy=accuracy)
            # print(prediction," x_test: ",X_test)
            record(prediction=prediction)
            decision_order = prediction[0]
            order_target_percent(context.security, decision_order)
    except Exception as error:
        print('Caught this error: ' + repr(error))


test_string = ['SPY']

start = pd.to_datetime('2009-01-01').tz_localize('US/Eastern')
end = pd.to_datetime('2018-12-01').tz_localize('US/Eastern')


try:
    for ele in test_string:
        SYMBOL = ele
        if(os.path.isfile(('output/'+SYMBOL+'_NEAT_output.csv'))):
            print('output/'+SYMBOL+'_NEAT_output.csv is exist')
            continue
        else:
            print('output/' + SYMBOL + '_NEAT_output.csv is not exist')
        perf_manual = run_algorithm(start = start, end = end, capital_base = 10000000.0,  initialize=initialize, handle_data=rebalance, bundle = stocks_bundle)

        # Print
        perf_manual.to_csv('output/'+SYMBOL+'_NEAT_output.csv')
except Exception as error:
    print(error)




