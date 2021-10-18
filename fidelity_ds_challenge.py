import os
import sys
import re
import platform
import itertools
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from sklearn.metrics import *
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from scipy.stats import uniform, loguniform

from prophet import Prophet

import lightgbm as lgb
import shap
import featuretools as ft

import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.seasonal import seasonal_decompose
from IPython.display import Markdown, display

pd.set_option('display.max_colwidth', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_row', 200)

print('Python version:', sys.version)

print('MAC OS version:', platform.mac_ver()[0])

# define utility functions
def convert_date(df, date_column, new_column='date', format='%m-%d-%Y', errors='coerce'):
    df[new_column] = pd.to_datetime(df[date_column], format=format, errors=errors)
    return df

def read_directory(path, nrows=None, columns=None):
    result = []
    files = list(os.listdir(path))
    for file in files:
        df = pd.read_csv(f'{path}/{file}', nrows=nrows, usecols=columns, index_col=False,
                         encoding='windows-1252')
        result += [df]
    result = pd.concat(result)
    return result

def clean_data(values):
    values = values.replace(['NaN', 'nan', 'NAN', 'null'], np.nan)
    try:
        values = values.astype(float)
    except:
        pass
    try:
        values = values.astype(str)
    except:
        pass
    return values


def add_date_features(df, date_features, date_column):
    # add date features based on SALEDATE before splitting universe (no lookahead bias since date info is a given)
    exclude_missing_sales = df[date_column].notnull()
    # inception date is the 100 years before the earliest in the whole dataset 
    inception_date = df[date_column].min() - pd.Timedelta(value=52*10, unit='W')

    for f in date_features:
        df[f] = np.nan
        sale_date = df.loc[exclude_missing_sales, date_column]
        if f == 'DAYSFROMINCEPTION':
            values = (sale_date - inception_date).dt.days
        elif f == 'MONTH':
            values = sale_date.dt.month
        elif f == 'WEEKDAY':
            values = sale_date.dt.dayofweek
        elif f == 'DAYOFYEAR':
            values = sale_date.dt.dayofyear
        elif f == 'YEAR':
            values = sale_date.dt.year
        else:
            date_features_str = ','.join(date_features)
            raise ValueError(f'Date feature must be in {date_features_str}!')
        df.loc[exclude_missing_sales, f] = values
    return df


# define universes and get filters
def define_universes(df, test_start_date, test_end_date, date_column='RETURNSTART', target_column='ANNUALRETURN'):
    exclude_missing_sales = df['SALEPRICE'].notnull()
    invalid_sale_code = ['1', '32', '2', 'BK', '3', '35', '13', '99']
    exclude_nonopen_sale_types = ~df['SALECODE'].isin(invalid_sale_code)
    exclude_invalid_sale_types = df['SALECODE'].isin(['0', 'U', 'UR'])
    is_dev = df[date_column] < test_start_date
    is_test = df[date_column].between(test_start_date, test_end_date)
    exclude_invalid_sale_dates = (df['SALEDATE'] - df['PREVSALEDATE']).dt.days > 0
    exclude_invalid_sale_dates &= (df['PREVSALEDATE'] - df['PREVSALEDATE2']).dt.days > 0

    # at training time, we filter more transactions with more data to reduce bias/variance 
    dev_data = df[is_dev & exclude_invalid_sale_types & exclude_invalid_sale_dates & exclude_missing_sales]
    target = dev_data[target_column]
    dev_data[target_column] = target.clip(target.quantile(0.1), target.quantile(0.9))
    dev_data = dev_data.dropna(subset=[target_column])

    test_data = df[is_test & exclude_invalid_sale_types & exclude_invalid_sale_dates & exclude_missing_sales]
    test_data[target_column] = test_data[target_column].replace(np.inf, np.nan).clip(target.quantile(0.1), target.quantile(0.9))
    test_data = test_data.dropna(subset=[target_column])
    return dev_data, test_data


def encode_sparse_cols(df):
    for col in ['PROPERTYUNIT', 'PROPERTYFRACTION']:
        mode = df[col].value_counts().sort_values().index[-1]
        df[col] = df[col].apply(lambda x: 1 if x == mode else 0)
    return df


def encode_ordinal_cols(df):
    grade_map = {
        'XX': 0, 
        'X+': 1,
        'X': 2,
        'X-': 3,
        'A+': 4,
        'A': 5,
        'A-': 6,
        'B+': 7,
        'B': 8,
        'B-': 9,
        'C+': 10,
        'C': 11,
        'C-': 12,
        'D+': 13,
        'D': 14,
        'D-': 15,
        'E+': 16,
    }

    condition_map = {
        1.0: 0,
        7.0: 1,
        2.0: 2,
        3.0: 3,
        4.0: 4,
        5.0: 5,
        6.0: 6,

    }

    cdu_map = {
        'EX': 0,
        'VG': 1,
        'GD': 2,
        'AV': 3,
        'FR': 4,
        'PR': 5,
        'VP': 6,
        'UN': 7,

    }
    df['GRADE'] = df['GRADE'].map(grade_map)
    df['CONDITION'] = df['CONDITION'].map(condition_map)
    df['CDU'] = df['CDU'].map(condition_map)
    return df


def perform_feature_engineering(data, float_cols, ordinal_cols, categorical_cols):
    for sale_date_col in ['SALEDATE', 'PREVSALEDATE', 'PREVSALEDATE2']:
        data = convert_date(data, date_column=sale_date_col, new_column=sale_date_col, format='%m-%d-%Y')

    # convert to float
    for c in float_cols:
        data[c] = pd.to_numeric(data[c], errors='coerce').astype(float)

    # convert ordinal features
    data = encode_ordinal_cols(data)
    from sklearn import preprocessing
    ordinal_col_encoders = {}
    for c in ordinal_cols:
        le = preprocessing.LabelEncoder()
        # convert to float for SHAP to work
        data[c] = le.fit_transform(data[c]).astype(float)
        ordinal_col_encoders[c] = le

    # convert categorical features to dummies
    data = encode_sparse_cols(data)
    dummies_cols = []
    cat_df_list = []
    for c in categorical_cols:
        # convert to str
        values = data[c].astype(str)
        cat_df = pd.get_dummies(values, prefix=c, drop_first=True) 
        # remove special characters from columns
        cat_df = cat_df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        cat_df = cat_df.loc[:, ~cat_df.columns.duplicated()]
        
        dummies_cols += cat_df.columns.tolist()
        cat_df_list += [cat_df]
        print('onehotencoding {} with {} categories'.format(c, cat_df.shape[1]))
    
    data = pd.concat([data] + cat_df_list, axis=1)
    return data, ordinal_col_encoders, dummies_cols


def compute_return(
    df,            
    initial_price='PREVSALEPRICE', 
    current_price='SALEPRICE', 
    initial_date='PREVSALEDATE', 
    current_date='SALEDATE', 
    suffix='',
):
    """assume income to be zero, not a great assumption, but we don't have income data"""
    # we cannot possibly think we will buy or sell for 1 dollar; exclude rows
    invalid_prices = [0., 1.]
    df['GROSSRETURN'+suffix] = df[current_price].replace(invalid_prices, np.nan) / \
                               df[initial_price].replace(invalid_prices, np.nan)
    df['HOLDINGPERIOD'+suffix] = (df[current_date] - df[initial_date]).dt.days / 365
    holding_period_return = df['GROSSRETURN'+suffix] ** (1./df['HOLDINGPERIOD'+suffix]) - 1
    df['RETURNSTART'+suffix] = df[initial_date]
    df['RETURNSTARTPRICE'+suffix] = df[initial_price]
    df['RETURNEND'+suffix] = df[current_date]
    df['ANNUALRETURN'+suffix] = holding_period_return
    return df


def add_target(df, index_level):
    # compute long-term return from PREVSALEPRICE2 to SALEPRICE
    df = compute_return(
        df,
        initial_price='PREVSALEPRICE2', 
        current_price='SALEPRICE', 
        initial_date='PREVSALEDATE2', 
        current_date='SALEDATE',
    )

    # compute short-term return from PREVSALEPRICE2 to SALEPRICE
    df = compute_return(
        df, 
        suffix='FALLBACK',
    )
    
    df['IMPUTE'] = np.nan
    # fill missing long-term returns with short-term return
    has_missing_return = df['ANNUALRETURN'].isnull()
    print('LONGTERM has missing {}% returns'.format(has_missing_return.mean()*100))
    df.loc[~has_missing_return, 'IMPUTE'] = 'LONGTERM'
    df.loc[has_missing_return, 'ANNUALRETURN'] = df['ANNUALRETURNFALLBACK'] 
    df.loc[has_missing_return, 'RETURNSTART'] = df['RETURNSTARTFALLBACK']
    df.loc[has_missing_return, 'RETURNSTARTPRICE'] = df['RETURNSTARTPRICEFALLBACK']
    df.loc[has_missing_return, 'RETURNEND'] = df['RETURNENDFALLBACK'] 
    df.loc[has_missing_return & df['ANNUALRETURNFALLBACK'].notnull(), 'IMPUTE'] = 'SHORTTERM'
    print('Filled {}% missing returns with SHORTTERM'.format((df['IMPUTE'] == 'SHORTTERM').mean()*100))
    
    # fill missing return with rolling 3-year market return of a median home 
    index_lookback_in_years = 3
    gross_index_return = index_level / index_level.shift(4*index_lookback_in_years)
    index_return = gross_index_return ** (1/index_lookback_in_years) - 1

    has_missing_return = df['ANNUALRETURN'].isnull()
    has_valid_dates = df['RETURNSTART'].notnull()
    prediction_start = df.loc[has_missing_return & has_valid_dates, 'RETURNSTART'].dt
    prediction_start = convert_to_index_quarterly_level(prediction_start)
    prediction_return = prediction_start.map(index_return.to_dict())
    df.loc[has_missing_return, 'ANNUALRETURN'] = prediction_return
    df.loc[has_missing_return & prediction_return.notnull(), 'IMPUTE'] = 'INDEX'
    print('Filled {}% missing returns with INDEX'.format((df['IMPUTE'] == 'INDEX').mean()*100))
    
    print('There are {}% missing returns left'.format(df['IMPUTE'].isnull().mean()*100))

    return df


def convert_to_index_quarterly_level(prediction_start):
    quarter_to_month = {1: 4, 2: 7, 3: 10, 4: 1}
    prediction_start = \
        prediction_start.year.astype(str) + prediction_start.quarter.map(quarter_to_month).astype(str) + '01'
    prediction_start = pd.to_datetime(prediction_start, format='%Y%m%d')
    return prediction_start


assessments = pd.read_csv('assessments/assessments-2021.csv', encoding='windows-1252')
assessments.head()


assessments_dict = pd.read_csv('property-assessments-data-dictionary.csv', encoding='windows-1252')
assessments_dict.head()

# ## expensive cell: only used in exploratory analysis

# # construct a dev sample for manual inspection 
# manual_dev_sample_size = 10000
# dev_data, _ = define_universes(assessments)
# # and get a sample for exploratory analysis
# rng = np.random.default_rng(1234)
# sample_idx = rng.choice(dev_data.shape[0], size=manual_dev_sample_size, replace=False)
# manual_dev_sample = dev_data.iloc[sample_idx]

# # explore each feature and classify it using dictionary description
# # break lines in description 
# n_char_per_line = 80
# for idx, field in enumerate(manual_dev_sample.columns.tolist()):
#     plt.figure(figsize=(10, 5))
#     label = assessments_dict.loc[assessments_dict['Field Name']==field, 'Field Description'].iat[0]
#     desc = assessments_dict.loc[assessments_dict['Field Name']==field, 'Additional Data Details'].iat[0]

#     n_lines = int(np.ceil(len(desc) / n_char_per_line))
#     legend_desc = [desc[i*n_char_per_line:(i+1)*n_char_per_line] for i in range(n_lines)]
#     legend_desc = '\n'.join(legend_desc)
#     plt.xticks(fontsize=10, rotation=60)

#     values = clean_data(manual_dev_sample[field])
#     clean_values = values.dropna()
#     n_unique = len(set(clean_values))
#     n_missing = values.shape[0] - clean_values.shape[0]

#     plt.hist(clean_values, bins=100, label=legend_desc)
#     plt.title(f"{field}: {label} with {n_unique} categories and {n_missing} NaN")
#     plt.legend(bbox_to_anchor=(0.8, -0.35))
#     plt.show()


# based on the observations from the histogram and the descriptions, I classify the cols in to these categories:
categorical_cols = ['PROPERTYCITY', 'TAXDESC', 'TAXSUBCODE', 'OWNERDESC', 'CLASS', 'USEDESC',
                    'HOMESTEADFLAG', 'FARMSTEADFLAG', 'CLEANGREEN', 'ABATEMENTFLAG', 'COUNTYEXEMPTBLDG',
                    'STYLEDESC', 'EXTFINISH_DESC', 'ROOF', 'BASEMENTDESC', 'GRADEDESC', 'HEATINGCOOLING', 
                    'CARDNUMBER', 'NEIGHDESC',  'MUNIDESC', 'SCHOOLDESC', 
                    # sparse features with 99% being a single value
                    'PROPERTYUNIT', 'PROPERTYFRACTION', 
                   ]
ordinal_cols = ['PROPERTYHOUSENUM', 'PROPERTYZIP', 'STORIES', 'YEARBLT',
                'TOTALROOMS', 'BEDROOMS', 'FULLBATHS', 'HALFBATHS', 'FIREPLACES', 'BSMTGARAGE', 
                # need to hard-code the ordinal map
                'GRADE', 'CONDITION', 'CDU', 
               ]
float_cols = ['LOTAREA', 'FINISHEDLIVINGAREA',
              'COUNTYBUILDING', 'COUNTYLAND', 'COUNTYTOTAL', 
              'LOCALBUILDING', 'LOCALLAND', 'LOCALTOTAL',
              'FAIRMARKETBUILDING', 'FAIRMARKETLAND', 'FAIRMARKETTOTAL']

test_start_date = dt.datetime(2016, 1, 1)
test_end_date = dt.datetime(2020, 11, 30)
# 1. add feature DAYSFROMINCEPTION to capture trend
# 2. add various feaures to capture price seasonality (e.g. MONTH, WEEKDAY)
date_features = ['DAYSFROMINCEPTION']
date_features += ['MONTH', 'WEEKDAY', 'DAYOFYEAR', 'YEAR']

# engineer features
assessments, ordinal_col_encoders, dummies_cols = perform_feature_engineering(
    assessments, float_cols, ordinal_cols, categorical_cols)

# get index level data to impute missing returns
index_level = pd.read_csv('pittsburg-quarterly-home-value-index.csv')
index_level = index_level.set_index('DATE').iloc[:, 0].rename('INDEXRETURN')
index_level.index = pd.to_datetime(index_level.index, format='%Y-%m-%d')

# add target variable expressed in annualized holding-period returns
assessments = add_target(assessments, index_level=index_level)

# add date features based on which date we use as return start date
assessments = add_date_features(assessments, date_features, date_column='SALEDATE')

dev_data, test_data = define_universes(assessments, test_start_date, test_end_date,
                                       date_column='SALEDATE', target_column='ANNUALRETURN')

# construct feature set
features = float_cols + ordinal_cols + dummies_cols + date_features

target = 'ANNUALRETURN'
y_dev = dev_data[target]
X_dev = dev_data[features]

# use recent 2 years as validation set
is_valid = dev_data['SALEDATE'] > '2014-01-01'
X_valid = dev_data.loc[is_valid, features]
X_train = dev_data.loc[~is_valid, features]
test_data[target] = test_data[target].replace(np.inf, np.nan)
test_data = test_data.dropna(subset=[target])
X_test = test_data[features]

y_valid = dev_data.loc[is_valid, target]
y_train = dev_data.loc[~is_valid, target]
y_test = test_data[target]

rs_params = {
        "num_boosting_rounds": loguniform(100, 10000),
        "num_leaves": (30, 100),
        "learning_rate": loguniform(1e-6, 1),
        "lambda_l1": loguniform(1e-6, 1),
        "lambda_l2": loguniform(1e-6, 1),
        'bagging_fraction': (0.5, 0.8),
        'bagging_frequency': (5, 8),
        "feature_fraction": (0.4, 0.8),
        "bagging_freq": [1, 5],
        "min_child_samples": (20, 1000),
    }

# # perform 5-fold CV on the training data
# rs_cv = RandomizedSearchCV(estimator=lgb.LGBMRegressor(objective="regression", silent=True),
#                            param_distributions=rs_params, cv=5, n_iter=10, verbose=0)
# rs_cv = rs_cv.fit(X_train, y_train)
# print(rs_cv.best_score_, rs_cv.best_params_)
# params = rs_cv.best_params_.copy()

# copy over the best parameter
params = {
     'bagging_fraction': 0.8,
     'bagging_freq': 1,
     'bagging_frequency': 8,
     'feature_fraction': 0.8,
     'lambda_l1': 1.129127174658804e-06,
     'lambda_l2': 0.00013476184921191474,
     'learning_rate': 0.07096990084956828,
     'min_child_samples': 1000,
     'num_boosting_rounds': 684.3926111589022,
     'num_leaves': 30
}
params['objective'] = 'regression'

print(params)
lgtrain = lgb.Dataset(X_train, label=y_train)
model = lgb.train(params, lgtrain, int(params['num_boosting_rounds']), verbose_eval=False)

# evaluation metrics
train_pred = model.predict(X_train)
valid_pred = model.predict(X_valid)
test_pred = model.predict(X_test)
print('train R2:', r2_score(y_train, train_pred))
print('valid R2:', r2_score(y_valid, valid_pred))
print('test R2:', r2_score(y_test, test_pred))

print('train rmse:', mean_squared_error(y_train, train_pred)**0.5)
print('valid rmse:', mean_squared_error(y_valid, valid_pred)**0.5)
print('test rmse:', mean_squared_error(y_test, test_pred)**0.5)

print(dev_data['IMPUTE'].fillna('missing').value_counts(True))
print(test_data['IMPUTE'].fillna('missing').value_counts(True))

# explain the model
explainer = shap.TreeExplainer(model)
rng = np.random.default_rng(1234)
sample_idx = rng.choice(X_train.shape[0], size=1000, replace=False)
sample_X_train = X_train.iloc[sample_idx]
shap_values = explainer.shap_values(sample_X_train)

shap.summary_plot(shap_values, sample_X_train, show=True)

for c in ['DAYSFROMINCEPTION', 'COUNTYLAND', 'YEARBLT', 'PROPERTYHOUSENUM', 'PROPERTYZIP', 'CONDITION']:
    shap.dependence_plot(c, shap_values, sample_X_train, show=True)


