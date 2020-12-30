##utility functions for preprocess data
'''
Description: translates ac360 to use case specific model dataset
Author: TEST
Creation date: May,2020
Last edited: July, 2020
'''

#native libraries
import pandas as pd
import numpy as np
import pickle
import gc
from collections import Counter
import json
from scipy import stats
import csv,argparse
import tabulate
from scipy import stats
#from util.io_layer import io_ds as io
import s3fs

##plotting libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
# import seaborn as sns

#sklean binarizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder

##s3 filereads
import s3fs
from pyarrow.filesystem import S3FSWrapper
import pyarrow.parquet as pq


## log / helper libraries
import datetime
from dateutil.relativedelta import relativedelta
from logger import logger

#DYNAMIC
today_dt = datetime.datetime.today().strftime('%Y-%m-%d')

def processing_pipeline(run_tag="run_tag",
                        run_date = today_dt,
                        id_columns = None,
                        holdout_months = 2,
                        time_field = 'monthyearid',
                        primary_target = 'target_flag',
                        primary_customer = 'customerseqid',
                        continuous_date_identifier = 'days_',
                        date_impute_strategy = "max"):
    '''
    Desc: Prepares base data for all use cases by performing series of preprocessing
    Params:
        *run_tag: unique label for each use case class
        *run_date: current date of run used for suffixing file path for training data and logging
        *id_columns: identifier columns e.g. customerseqid, monthyeard, target_flag etc.
        *holdout_months: # of months to be held out for clean testing
        *time_field: field containing date/month separator (default: monthyearid)
        *primary_target: target flag
        *primary_customer: customerseqid
        *continuous_date_identifier: prefix for date derived continuous fields (default: days_)
        *date_impute_strategy: imputation logic for temporal fields such as days_since. values available:
            min, max, mean or zeroes. to be entered as string
        '''

    #read training data from s3 parquet file
    try: # TODO check if you need date on training data # TODO dropping here temp
        dataframe = io.get(f"training_{run_tag}").drop(columns=['hid'])
    except:
        logger.critical("ERRROR: Please check the filepath for training data")
        raise

    #extract last n months for holdout data
    out_of_time_window = dataframe.loc[:, [time_field]].drop_duplicates()\
                    .sort_values(time_field, ascending = False).head(holdout_months) #limiting to latest n months

    #max month for scoring
    scoring_window = dataframe.loc[:, [time_field]].drop_duplicates()\
                    .sort_values(time_field, ascending = False).head(1) #limiting to latest n months

    dataframe['for_scoring'] = np.where(dataframe[time_field].isin(scoring_window[time_field]),1,0)
    logger.info("Added scoring tag for month {0}".format(list(scoring_window[time_field])))

    dataframe['holdout'] = np.where(dataframe[time_field].isin(out_of_time_window[time_field]),1,0)
    logger.info("Added holdout tag for months {0}".format(list(out_of_time_window[time_field])))
    logger.info("Split of data passed to core vs holdout= \n***********\n{0}"\
                .format(dataframe.holdout.value_counts()))
    ## TODO Piyush drop
    # dropping cols with 0 variance
    no_var_cols = no_variance_cols(dataframe)
    logger.info('Dropping following features because of 0 variance \n***********\n{0}'.format(list(no_var_cols)))
    dataframe = dataframe.drop(no_var_cols, axis = 1)
    logger.info('Columns with no variance removed \n***********\ntotal rows = {0} and total cols = {1} after removing 0 variance fields'\
                .format(dataframe.shape[0], dataframe.shape[1]))

    #fixing any inf/text NaN vals
    dataframe = dataframe.replace([np.inf, -np.inf, 'NA', 'NaN', 'others', 'missing'], np.nan)
    logger.info('Reformatted missing inf related entries to nulls')

    ## TODO Piyush fix + eom bom
    #dropping all datetime columns (if any)
    dt_cols = list(dataframe.select_dtypes(include= 'datetime64').columns.values)
    try:
        dataframe = dataframe.drop(dt_cols, axis = 1)
        logger.info('{0} date time cols found to be removed. \n***********\nList here: {1}'\
                    .format(len(dt_cols), list(dt_cols)))
    except:
        dataframe = dataframe
        logger.info('WARNING: No datetime cols found to be removed')

    #select object columns

    cols_to_encode = list(dataframe.select_dtypes(include = 'object').columns)
    cols_to_encode = [e for e in cols_to_encode if e not in id_columns]
    logger.info('Running encoding for seleced columns \n***********\n{0}'.format(cols_to_encode))

    #impute missing values with "missing" for label encodes
    dataframe.loc[:, cols_to_encode] = dataframe.loc[:, cols_to_encode].fillna("missing")

    #instantiate one hot encoder
    dataframe = pd.concat([dataframe.drop(cols_to_encode, axis=1),
                           pd.get_dummies(dataframe.loc[:, cols_to_encode])], axis=1)
    logger.info("total rows = {0} and total cols = {1} after dummy encoding"\
                .format(dataframe.shape[0], dataframe.shape[1]))

    #cols for max impute
    #selecting non date, continuous columns
    max_impute = [col for col in dataframe.columns if continuous_date_identifier in col]

    #impute date based features using max gap observed
    if date_impute_strategy == "max":
        dataframe.loc[:, max_impute] = dataframe.loc[:, max_impute].fillna(dataframe.max(axis=0))
    elif date_impute_strategy == "min":
        dataframe.loc[:, max_impute] = dataframe.loc[:, max_impute].fillna(dataframe.min(axis=0))
    elif date_impute_strategy == "mean":
        dataframe.loc[:, max_impute] = dataframe.loc[:, max_impute].fillna(dataframe.mean(axis=0))
    elif date_impute_strategy == "zeroes":
        dataframe.loc[:, max_impute] = dataframe.loc[:, max_impute].fillna(0)
    else:
        logger.warn('ERROR: Wrong imputation logic provided for date based continuus fields')



    logger.info('Total rows = {0} and total cols = {1} after max imputation'\
                .format(dataframe.shape[0], dataframe.shape[1]))

    #for all remaining cols impute with 0
    dataframe = dataframe.fillna(0)
    logger.info('Total rows = {0} and total cols = {1} after remaining continuous imputation'\
                .format(dataframe.shape[0], dataframe.shape[1]))

    #drop all columns that are not intuitive
    cols_all = list(dataframe.columns)
    '''
    todo
    '''
    #todo: check _missing (suffix)
    cols_drop = [text for text in cols_all if any(k in text.upper() for k in ['MISSING','OTHER','_NA'])]
    logger.info("Dropping columns that are not business explanatory\n***********\n{0}".format(cols_drop))
    dataframe = dataframe.drop(cols_drop, axis = 1)

    return dataframe

def no_variance_cols(dfname, cutoff = 1):
    '''
    dfname: name of input dataframe
    cutoff: cutoff of number of uniqe cols values that need to be dropped. default = 1: 0 variance)
    '''
    print('Total unique cols in the dataframe: {0}'.format(len(list(dfname.columns))))
    cat_cols = pd.DataFrame(dfname.nunique()).reset_index()
    cat_cols.columns = ['cols', 'cnt']

    #holding cols with 1 or less unique vals
    issue_cols_list = list(cat_cols.loc[cat_cols.cnt <= cutoff, 'cols'].values)
    logger.info("Total columns with no variance: {0}"\
            .format(len(issue_cols_list)))
    return issue_cols_list

def remove_id_cols(dfname, keywords):
    '''
    dfname: name of input dataframe
    keywords: comma separated list of keywords to be looked for in the column names for a match )
    '''
    search_kws = '|'.join(keywords)
    spike_cols = list(dfname.loc[:, dfname.columns.str.contains(search_kws)].columns)
    print('Total id cols matched: {0}'.format(len(spike_cols)))
    return spike_cols

def cat_cont_list_for_encoding(dfname, unique_val_cutoff = 30):
    '''
    dfname: name of input dataframe
    unique_val_cutoff:minimum unique values in the column for creating boundary between continuous and categorical feature
    '''

    print('Total unique cols in the dataframe: {0}'.format(len(list(dfname.columns))))
    cat_cols = pd.DataFrame(dfname.nunique()).reset_index()
    cat_cols.columns = ['cols', 'cnt']

    #Catg columns list for further analysis
    cat_cols_list = list(cat_cols.loc[(cat_cols.cnt < unique_val_cutoff) & (cat_cols.cnt > 1), 'cols'])
    cat_cols_list = list(set(cat_cols_list + list(dfname.select_dtypes(include = "object").columns)))

    #cont cols list for further analysis
    cont_cols_list= list(dfname.select_dtypes(include = [np.number]).columns.values)

    return cat_cols_list, cont_cols_list

def s3_parquet_to_pandas_local(s3_file_path):
    '''
    helper function to read parquet file and translate to pandas
    '''
   ##read core files from parquet folder
    df = pd.read_parquet(s3_file_path)
    logger.info("Shape of data loaded: {0}".format(df.shape))
    return df

def s3_parquet_to_pandas(s3_file_path):
    '''
    helper function to read parquet file and translate to pandas
    '''
    ##read core files from parquet folder
    fs = s3fs.S3FileSystem()
    # fileread
    dataset = pq.ParquetDataset(s3_file_path,filesystem=fs)
    table = dataset.read()
    df = table.to_pandas()
    logger.info("Shape of data loaded: {0}".format(df.shape))
    return df

def s3_csv_to_pandas(s3_file_path):
    '''
    helper function to read csv file from s3 file and translate to pandas
    '''
    df = pd.read_csv(s3_file_path)
    logger.info("Shape of data loaded: {0}".format(df.shape))
    return df

def pandas_to_s3_csv(df, out_path = 'psm/test.csv'):
    '''
    helper function to read csv file from s3 file and translate to pandas
    '''
    df.to_csv('s3://data-analytics-emr-data-sit-income-com-sg/{0}'.\
              format(out_path), index=False)
    logger.info("Shape of data exported: {0}".format(df.shape))
    return None

##utility functions for transforming processed ac360 data into model inputs
'''
Description: transforms processed ac360 data pertaining to a use case into model input data using a series of user controlled steps (such as feature selection, correlation checks and train/test/holdout splits
Author: TEST
Creation date: May,2020
Last edited: Sep, 2020
'''

#native libraries
import pandas as pd
import numpy as np
import pickle
import gc
import os
from collections import Counter
import json
from scipy import stats
import csv,argparse
import tabulate
from scipy import stats

#sklearn specific modules
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
# from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.metrics import auc, roc_auc_score, roc_curve, f1_score, mean_squared_error, confusion_matrix,make_scorer, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFE

#external model libraries
# from xgboost.sklearn import XGBClassifier
# from lightgbm import LGBMClassifier
# from bayes_opt import BayesianOptimization
# from catboost import CatBoostClassifier
# from skopt.space import Real, Categorical, Integer
# from skopt import BayesSearchCV
# from bayes_opt import BayesianOptimization
from imblearn.over_sampling import SMOTE

##plotting libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
# import seaborn as sns

## log / helper libraries
import datetime
from dateutil.relativedelta import relativedelta
from logger import logger

def model_data_generator_sampling(dataframe,
                                target_label,
                                primary_customer,
                                time_key = 'monthyearid',
                                id_cols = None,
                                today_dt = None,
                                n_feats_REF = 300,
                                seed = 42,
                                holdout_column = "holdout",
                                scoring_data_column = "for_scoring",
                                downsampling = None,
                                exclude_cols = [],
                                rfe_test_size = 0.5,
                                model_test_size = 0.3,
                                corr_threshold = 0.85,
                                feature_export_path = None):

    '''
    desc: creates all cuts of train and test data for feeding into model pipeline
    inputs:
        *dataframe: starting dataframe with all featues
        *target_label: primary target field (e..g target_flag)
        *primary_customer: primary key of the dataframe (e.g. customerseqid)
        *seed: seed values
        *holdout_column: binary field for holdout identification
        *id_cols: list of identifier columns
        *today_dt: run date, default to todays date
        *n_feats_REF: number of features to be defaulted for selecting from REF
        *exclude_cols: columns not be deleted during feature selection
        *rfe_test_size: (1-data that needs to be passed for RFE estimation)
        *model_test_size: ratio to be separated out for model testing
        *corr_threshold: threshold for feature correlation. all features above this val to be excluded
    '''

    #Back up base data for transformations
    model_df = dataframe.copy()
    logger.info("Total rows = {0} and total cols = {1} after preprocessing"\
            .format(model_df.shape[0], model_df.shape[1]))

    #Build small train/test for rfe estimation
    rfe_train, rfe_test = train_test_split(model_df, test_size = rfe_test_size, random_state = seed)
    rfe_x = rfe_train.drop(labels=id_cols, axis=1)
    rfe_y = rfe_train[target_label]

    #Create folder for exporting features
    if feature_export_path:
        out_path_local = '{0}{1}/features/'.format(feature_export_path, today_dt)
        if not os.path.exists(out_path_local):
            os.makedirs(out_path_local)
        feat_file = '{0}feat_RFE_{1}.csv'.format(out_path_local, today_dt)
        logger.info('Exporting Features post RFE at {0}'.format(feat_file))

    ##run feature selection here
    feat_out  = feat_ranking_RFE(rfe_x,rfe_y,\
                              feat_file,seed=seed)

    #Perform Feature selection via RFE

    RFE_selected_feats = feat_selector_RFE(n_feats_REF, check_cols = exclude_cols,
        load_feat_filename = feat_file)

    #Prepare subset of rfe dataset to only include features selected from RFE
    X_train_cor = rfe_x[RFE_selected_feats]
    high_cor_df, remove_feat = \
    feat_selector_correlation_pandas(X_train_cor, exclude_cols, threshold = corr_threshold)
    # TODO remove
    selected_feats = list(set(RFE_selected_feats)-set(remove_feat))
    logger.info("{0} features selected after filtering out highly correlated features: \n ------------- \n{1}"\
                .format(len(selected_feats), selected_feats))


    #adding id cols back to the list of columns
    selected_feats  = selected_feats + id_cols
    model_df = model_df.loc[:, selected_feats]

    #check na values
    model_df = model_df.dropna(axis = 1)
    logger.info("Total rows = {0} and total cols = {1} after rfe reduction"\
                .format(model_df.shape[0], model_df.shape[1]))

    ##renaming cols for lightgbm specifically
    model_df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in model_df.columns]

    #Export final column names
    col_pickle_path = '{0}{1}/features/final_model_columns.pkl'.format(feature_export_path, today_dt)
    with open(col_pickle_path, 'wb') as f:
        pickle.dump(list(model_df.columns), f)
    logger.info('Final features saved at {0}'.format(col_pickle_path))

    ##holdout for final validation and removing holdout from modeling data
    holdout_df = model_df.loc[model_df[holdout_column] == 1]

    ##extracting latest month for scoring purposes
    for_scoring_df = model_df.loc[model_df[scoring_data_column] == 1]

    #passing rest to train
    model_df = model_df.loc[model_df[holdout_column] == 0]




    #Create train test splits for core model
    train_original, test_original = \
        train_test_split(model_df, test_size = model_test_size, random_state = seed)

    #Extract test data
    X_test=test_original.drop(labels=id_cols, axis=1)
    y_test=test_original[target_label]

    if downsampling == 1:
        train_sample= custom_downsampling(train_original, primary_customer, target_label, time_key)
    else:
        train_sample= train_original

    ##Prepae train/test dataframes
    X_train=train_sample.drop(labels=id_cols, axis=1)
    y_train=train_sample[target_label]

    #Log summary info

    logger.info("\n*****************\nshape for model_df = {0} \n \
                shape for X_train = {1} \n \
                shape for X_test = {2} \n \
                shape for y_train = {3} \n \
                shape for y_test = {4} \n \
                shape for holdout_df = {5} \n \
                shape for train_original = {6} \n \
                shape for test_original = {7} \n \
                shape for train_sample = {8} \n \
                shape for scoring data = {9} \n\
                ".format(model_df.shape,
                        X_train.shape,
                        X_test.shape,
                        y_train.shape,
                        y_test.shape,
                        holdout_df.shape,
                        train_original.shape,
                        test_original.shape,
                        train_sample.shape,
                        for_scoring_df.shape))

    return X_train, X_test, y_train, y_test, holdout_df, for_scoring_df

def custom_downsampling(dataframe, customer_key, target_key, stratification_key):
    '''
    #desc: downsamples the repeated-month data to increase event rate by retaining
    1 record per non purchser and  only the purchase record of customers with target purchases
    #inputs:
        *dataframe: dataframe at customer-month level granularity
        *customer_key: identifer for customerseqid
        *target_key: binary variable (1/0) capturing purchase (e.g. target_label)
        *stratification_key: variable to sample customers across (e.g. monthyearid)
        '''
    #instantiate the run
    logger.info('Input params for downsampling are customer_key {0}, target_key {1}, stratification_key {2}'.format(customer_key, target_key, stratification_key))

    #filter for purchasers only
    purchasers_df = dataframe.loc[dataframe[target_key] == 1]

    #retain non purchasers' data only
    nonpurchasers_df = dataframe.loc[~dataframe[customer_key].isin(purchasers_df[customer_key])]

    #randomly sample 1 record per non purchaser
    nonpurchasers_df = nonpurchasers_df.sample(frac = 1.0).groupby(customer_key).head(1)

    #concat data for purchasers and non purchasers
    df_out = pd.concat([purchasers_df, nonpurchasers_df], axis = 0)

    ##extract summary for original data
    tdf_old = dataframe.groupby([stratification_key, target_key])[[customer_key]].count().reset_index()\
            .rename({customer_key : 'original_count'}, axis = 1)

    tdf_old = pd.pivot_table(tdf_old, values='original_count', index=[stratification_key],
                    columns=[target_key], aggfunc=np.sum).reset_index()\
                    .rename({0 : 'nonpurch_orig',
                            1: 'purch_orig'}, axis = 1)

    ##extract summary for sampled info
    tdf_new = df_out.groupby([stratification_key, target_key])[[customer_key]].count().reset_index()\
            .rename({customer_key : 'sampled_count'}, axis = 1)


    tdf_new = pd.pivot_table(tdf_new, values='sampled_count', index=[stratification_key],
                    columns=[target_key], aggfunc=np.sum).reset_index()\
                    .rename({0 : 'nonpurch_sampled',
                                            1: 'purch_sampled'}, axis = 1)

    #merging original and sampled info for comparison
    tdf_new = tdf_old.merge(tdf_new, on = [stratification_key], how = "left")

    ##adding event rates
    tdf_new['original_eventrate'] = tdf_new.purch_orig/tdf_new.loc[:, \
                                                                           ['purch_orig',
                                                                           'nonpurch_orig']].sum(axis = 1)

    tdf_new['sampled_eventrate'] = tdf_new.purch_sampled/tdf_new.loc[:, \
                                                                       ['purch_sampled',
                                                                       'nonpurch_sampled']].sum(axis = 1)
    print(tdf_new['original_eventrate'])
    print(tdf_new['sampled_eventrate'])
    tdf_new['samp_lift'] = (tdf_new['sampled_eventrate']/ tdf_new['original_eventrate']).astype(int)

    #display(tdf_new)
    logger.info("Summary of sampled data \n ------------- \n {0}".format(tdf_new))
    return df_out


def feat_selector_correlation_pandas(input_pandas_df, exclude_list, threshold, only_load = False, logging=False, filename=""):
    """
    desc: One of feature selection techniques (only for numeric features), which generate the features to be removed due to the highly Correlation. It is the python implementation for the function of feat_selector_correlation.
    inputs:
        * input_pandas_df: the input pandas dataframe.
        * exclude_list (list of str): list of columns which are excluded from the calculation of correlation coefficients.
        * threshold: the threshold for the correlation coefficients defined as highly correlated.
        * only_load (Boolean, optional): True if just only load the output remove_feat from the pickle file (the path is specified in filename).
          NOTE:
            1. There are randomness in the feature dropping (defined in num_feat_trimming), the remove_feat need to be saved for the process replication.
            2. When only_load = True, the calculation of correlation coefficients will still be carried out and return high_cor_df
        * logging (Boolean, optional): flag to indicate whether save the resulted remove_feat.
        * filename (str, optional): the path where the resulted remove_feat is saved to if logging = True.
    returns:
        * high_cor_df: the pandas dataframe which store the pair of features that are highly correlated and the correspounding correlation coefficients.
        * remove_feat (list of str): the features to be removed due to the highly Correlation.
    """
    check_logging_to_file("the resulted remove_feat", logging, filename)
    input_df = input_pandas_df.drop(exclude_list, axis=1)

    corr_coeff = input_df.corr(method='pearson')
    logger.info("Complete the correlation calculation!")
    cor_list = pandas_to_list(corr_coeff)
    corr_abs = pd.DataFrame(cor_list, columns=['fea1','fea2','coeff'])
    corr_abs = corr_abs[~corr_abs['coeff'].isna()]
    corr_abs['coeff'] = corr_abs['coeff'].astype('float').abs()

    high_cor_df = corr_abs[(corr_abs['coeff']>=threshold) & (corr_abs['coeff']<1)]

    if only_load:
        if filename =="":
            logger.warning("Please specify the file path for the features highly correlated and need to be removed.")
        else:
            logger.info("Loading the remove_feat from pickle file {0}...".format(filename))
            remove_feat = load_pickle(filename)
    else:
        remove_feat = num_feat_trimming(high_cor_df)
        if logging:
            to_pickle(remove_feat, filename)
            logger.info("Saved to the remove_feat to pickle file: {0}".format(filename))

    logger.info("{0} features in the highly correlated list (to be removed): {1}".format(len(remove_feat), remove_feat))
    return  high_cor_df, remove_feat


def feat_ranking_RFE(X_train,y_train,filename,seed=101):
    """
    desc: One of feature selection techniques for python implementation, called recursive feature elimination (RFE).
    Note that this technique is not available for spark. The function is to generate the feature importance file.
    inputs:
        * X_train (pandas dataframe): the dataframe with features for modeling.
        * y_train (pandas dataframe): the dataframe with label for modeling.
        * filename (str, optional): the path where the resulted feature importance from RFE is saved to.
        * seed (int, optional): the seed for the function.
    returns:
        * None
    """
    check_logging_to_file("the resulted feature importance", True, filename)
    logger.info("Start the feature selection (RFE)...")
    estimator = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=seed)
    selector = RFE(estimator, step=10)
    selector = selector.fit(X_train,y_train)
    logger.info("Complete the feature selection (RFE)!")

    ls_feature_importance = []
    for idx in range(len(X_train.columns)):
        ls_feature_importance.append([X_train.columns[idx], selector.ranking_[idx]])
    df_feature_importance = pd.DataFrame(data=ls_feature_importance, columns=['feature','ranking'])
    df_feature_importance = df_feature_importance.sort_values('ranking', ascending=True)
    df_feature_importance.to_csv(filename)
    logger.info("Complete the feature importance file saving to {0}".format(filename))

    return df_feature_importance

def feat_selector_RFE(n_feats, check_cols=None, exclude_cols=None, load_flag=True,
                X_train=None,y_train=None,seed=101,logging=False,load_feat_filename=""):
    """
    desc: One of feature selection techniques for python implementation, called recursive feature elimination (RFE).
    Note that this technique is not available for spark.
    inputs:
        * n_feats: number of features to be selected.
        * check_cols (list of str, optional): list of columns/features, which is checked whether is in the selected feature list.
        * exclude_cols (list of str, optional): list of columns/features, which is excluded from the selected feature list.
        * load_flag (Boolean, optional): True if loading the existing feature importance file generated by RFE
        via the function of feat_ranking_RFE.
        * X_train (pandas dataframe, optional): the dataframe with features for modeling. It is only required when runing RFE (load_flag=False).
        * y_train (pandas dataframe, optional): the dataframe with label for modeling. It is only required when runing RFE (load_flag=False).
        * seed (int, optional): the seed for the function.
        * logging (Boolean, optional): flag to indicate whether save the resulted feature importance from RFE.
        * load_feat_filename (str, optional): the path where the resulted feature importance from RFE is saved to if logging = True
        or generated by the function of feat_ranking_RFE.
    returns:
        * select_feats (list of str): the list of selected features via RFE.
    """
    if load_flag:
        df_feature_importance=pd.read_csv(load_feat_filename)
    else:
        logger.info("Start the feature selection (RFE)...")
        estimator = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=seed)
        selector = RFE(estimator, 1, step=1)
        selector = selector.fit(X_train,y_train)
        logger.info("Complete the feature selection (RFE)!")

        ls_feature_importance = []
        for idx in range(len(X_train.columns)):
            ls_feature_importance.append([X_train.columns[idx], selector.ranking_[idx]])
        df_feature_importance = pd.DataFrame(data=ls_feature_importance, columns=['feature','ranking'])
        df_feature_importance = df_feature_importance.sort_values('ranking', ascending=True)
        # save the feature importance
        check_logging_to_file("the resulted feature importance", logging, load_feat_filename)
        if logging:
            df_feature_importance.to_csv(load_feat_filename)

    # exclude the columns in exclude_cols in the selected feature list
    if exclude_cols!=None:
        df_feature_importance = df_feature_importance[~df_feature_importance['feature'].isin(exclude_cols)]
    df_feature_importance = df_feature_importance.reset_index(drop=True)
    select_feats = list(df_feature_importance.loc[:(n_feats-1),'feature'])
    logger.info("{0} features been selected: {1}".format(len(select_feats),select_feats))
    # check whether the defined feature list check_cols been selected
    if check_cols!=None:
        in_list = [column for column in check_cols if column in select_feats]
        out_list = [column for column in check_cols if column not in select_feats]
        logger.info("Out of features to check {0}, these are features are be selected: {1}".\
                   format(check_cols, in_list))
        if len(out_list)!=0:
            logger.warning("These are features not be selected: {0}".format(out_list))
        else:
            logger.info("There is no features from check_cols are not be selected.")
    return select_feats

'''
def feat_selector_RF(X_train, y_train, n_feats,logging=False, feat_imp_filename="",seed=101):
    """
    desc: One of feature selection techniques for python implementation, using Random Forest modeling. This technique is to run the Random Forest modelling algorithm with full set of features and then select the top N features from the feature importance list generated by the trained Random Forest model.
    inputs:
        * X_train (pandas dataframe): the dataframe with features for modeling.
        * y_train (pandas dataframe): the dataframe with label for modeling.
        * n_feats: number of features to be selected.
        * logging (Boolean, optional): flag to indicate whether save the resulted feature importance from RF.
        * feat_imp_filename (str, optional): the path where the resulted feature importance from RF is saved to if logging = True.
        * seed (int, optional): the seed for the function.
    returns:
        * selected_feats_RF (list of str): the list of selected features via RF.
    """
    rf = RandomForestClassifier(random_state=seed)
    PARAMETER_GRID = {
                    "RF": {
                        "param_grid": [{'n_estimators': [500],
                                        'max_leaf_nodes':[10,20,50],
                                       'max_depth': [7,10,20]}],
                        "model":rf
                    }
    }

    grid_search_rf = util.grid_search_model("RF",PARAMETER_GRID,X_train,y_train,CROSSVAL_N=3)
    sel_model = grid_search_rf.best_estimator_
    feat_imp = util.evaluate(sel_model, X_train, y_train)
    if logging:
        feat_imp.to_csv(feat_imp_filename)
        logger.info("The feature importance file (feature selection model RF) is saved to: {0}".format(feat_imp_filename))
    # select top n features based on RF model feature importance
    selected_feats_RF = [str(feat) for feat in list(feat_imp['features'][:n_feats])]
    return selected_feats_RF
'''

def check_logging_to_file(subject_name, logging, path):
    """
    desc: check whether the logging inputs are valid.
    inputs:
        * subject_name (str): the subject name which is saved, for the printing/logging purpose.
        * logging (Boolean): the flag to indicate whether the logging is enabled
        * path (str): the path which the variable is saved to.
    returns:
        * None
    """
    if not logging:
        logger.info("No logging is enabled.")
    elif path =="":
        raise ValueError('logging set to True, but no path provided')
    else:
        logger.info("Logging is enabled and {0} will save to: {1}".format(subject_name, path))

def feat_selector_correlation_pandas(input_pandas_df, exclude_list, threshold, only_load = False, logging=False, filename=""):
    """
    desc: One of feature selection techniques (only for numeric features), which generate the features to be removed due to the highly Correlation. It is the python implementation for the function of feat_selector_correlation.
    inputs:
        * input_pandas_df: the input pandas dataframe.
        * exclude_list (list of str): list of columns which are excluded from the calculation of correlation coefficients.
        * threshold: the threshold for the correlation coefficients defined as highly correlated.
        * only_load (Boolean, optional): True if just only load the output remove_feat from the pickle file (the path is specified in filename).
          NOTE:
            1. There are randomness in the feature dropping (defined in num_feat_trimming), the remove_feat need to be saved for the process replication.
            2. When only_load = True, the calculation of correlation coefficients will still be carried out and return high_cor_df
        * logging (Boolean, optional): flag to indicate whether save the resulted remove_feat.
        * filename (str, optional): the path where the resulted remove_feat is saved to if logging = True.
    returns:
        * high_cor_df: the pandas dataframe which store the pair of features that are highly correlated and the correspounding correlation coefficients.
        * remove_feat (list of str): the features to be removed due to the highly Correlation.
    """
    check_logging_to_file("the resulted remove_feat", logging, filename)
    input_df = input_pandas_df.drop(exclude_list, axis=1)

    corr_coeff = input_df.corr(method='pearson')
    logger.info("Complete the correlation calculation!")
    cor_list = pandas_to_list(corr_coeff)
    corr_abs = pd.DataFrame(cor_list, columns=['fea1','fea2','coeff'])
    corr_abs = corr_abs[~corr_abs['coeff'].isna()]
    corr_abs['coeff'] = corr_abs['coeff'].astype('float').abs()

    high_cor_df = corr_abs[(corr_abs['coeff']>=threshold) & (corr_abs['coeff']<1)]

    if only_load:
        if filename =="":
            logger.warning("Please specify the file path for the features highly correlated and need to be removed.")
        else:
            logger.info("Loading the remove_feat from pickle file {0}...".format(filename))
            remove_feat = load_pickle(filename)
    else:
        remove_feat = num_feat_trimming(high_cor_df)
        if logging:
            to_pickle(remove_feat, filename)
            logger.info("Saved to the remove_feat to pickle file: {0}".format(filename))

    logger.info("{0} features in the highly correlated list (to be removed): {1}".format(len(remove_feat), remove_feat))
    return  high_cor_df, remove_feat

def pandas_to_list(input_pandas):
    input_pandas_dict = input_pandas.to_dict('index')
    output_list = []
    for key1, value1 in input_pandas_dict.items():
        for key2, value2 in value1.items():
            row_list = [key1, key2, value2]
            output_list.append(row_list)
    return output_list

def num_feat_trimming(input_df):
    """
    desc: The function will trim numeric features which are highly correlated, and generate the list of features to be removed based on:
    <todo: add in rules for feature selection!>
    inputs:
        * input_df: the input spark dataframe to be strimmed.
    returns:
        * remove_list: the list of features to be removed.
    """

    # generate the feature clusters, which each features are highly correlated with all the other features
    cor_list = []
    for index, row in input_df.iterrows():
        cor_cluster = [row['fea1'], row['fea2']]
    #     print cor_cluster
        last_element = []

        while cor_cluster[-1] != last_element:
            # get the last_element before the iteration
            last_element = cor_cluster[-1]

            list_for_last_element = input_df[input_df['fea1']==last_element]['fea2'].tolist()

            # last element always correlated with all the items in cor_cluster
            list_for_last_element = list(set(list_for_last_element)-set(cor_cluster))

            #for rem_item in cor_cluster:
            #    if rem_item != last_element:
            #        list_for_last_element.remove(rem_item)

            if list_for_last_element !=[]:
                for item in list_for_last_element:
                    # if the new element is correlated to all the elements in cor_cluster
                    item_list = input_df[input_df['fea1']==item]['fea2'].tolist()
                    if len(item_list) ==len(set(item_list+cor_cluster)):
                        cor_cluster.append(item)
                        break
                    #if all(elem in input_df[input_df['fea1']==item]['fea2'].tolist() for elem in cor_cluster):
                    #    cor_cluster.append(item)
                    #    break

        cor_list.append(cor_cluster)
    unique_set_list = []
    for item in cor_list:
        itemset = set(item)
        if itemset not in unique_set_list:
            unique_set_list.append(itemset)

    unique_list = []
    for item in unique_set_list:
        itemlist = list(item)
        unique_list.append(itemlist)

    unique_list_union = []
    for item_list in unique_list:
        for item in item_list:
            unique_list_union.append(item)
    feat_freq = Counter(unique_list_union)
    # feat_freq = pd.DataFrame(sorted(feat_freq.items(), key=lambda x: x[1], reverse=True),columns=['feat', 'count'])

    selected_feat_list = []
    for each_cluster in unique_list:
        candidate = []
        candidate2 = dict()
        final_candicate = dict()

        for item in each_cluster:
            if not any(x in item for x in ['mob_avg_', 'tv_avg_', 'fl_avg_', 'bb_avg_', 'delta_', 'ratio_']):
                candidate.append(item)

        if candidate ==[]:
            derive_key = 0
            derive_list = []
            for derived_item in each_cluster:
                if derived_item.startswith('delta_') | derived_item.startswith('ratio_'):
                    derive_list.append(derived_item)
                    derive_key = 1
                candidate2[derived_item] = feat_freq[derived_item]

            if derive_key == 1:
                for derive_item in derive_list:
                    final_candicate[derive_item] = candidate2[derive_item]
            else:
                final_candicate = candidate2

            selected = sorted(final_candicate.items(), key=lambda x: x[1], reverse=True)[0][0]
        else:
            basic_key = 0
            a3m_key = 0
            basic_list = []
            a3m_list = []
            for cand_item in candidate:
                if not any(x in cand_item for x in ['a6m_', 'a3m_']):
                    # take the one based on frequency
                    basic_list.append(cand_item)
                    basic_key = 1
                elif cand_item.startswith('a3m_'):
                    a3m_list.append(cand_item)
                    a3m_key = 1
                    # rest will be a6m_
                candidate2[cand_item] = feat_freq[cand_item]

            if basic_key == 1:
                for basic_item in basic_list:
                    final_candicate[basic_item] = candidate2[basic_item]
            elif a3m_key == 1:
                for a3m_item in a3m_list:
                    final_candicate[a3m_item] = candidate2[a3m_item]
            else:
                final_candicate = candidate2

            selected = sorted(final_candicate.items(), key=lambda x: x[1], reverse=True)[0][0]
        selected_feat_list.append(selected)
        remove_list = list(set(input_df['fea1'].tolist())-set(selected_feat_list))
    return remove_list

'''
Description: runs a series of model training and throws different summary stats for the model performance
Author: TEST
Creation date: May,2020
Last edited: Sep, 2020
'''

#native libraries
import pandas as pd
import numpy as np
import pickle
import gc
import os
from collections import Counter
from collections import OrderedDict
import json
from scipy import stats
import csv,argparse
import tabulate
from scipy import stats

#sklearn specific modules
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
# from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.metrics import auc, roc_auc_score, roc_curve, f1_score, mean_squared_error, confusion_matrix,make_scorer, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFE

#external model libraries
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization
from catboost import CatBoostClassifier
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV
from bayes_opt import BayesianOptimization
from imblearn.over_sampling import SMOTE

##plotting libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
# import seaborn as sns

## log / helper libraries
import datetime
from dateutil.relativedelta import relativedelta
from logger import logger
#from util.io_layer import io_ds as io


def bayes_model_core(X_train, X_test, y_train, y_test, today_dt,
                     run_tag = 'base_model', est = 'XGB', ITERATIONS = 10,
                     seed= 42, verbosity = 10, refit_flag = False, cv_splits = 3,
                     calibration_method=None):
    '''
    desc: performs bayesian search per model specified by the user and performs base nfold cv to identify best model config
    
    input:
        *X_train: input training dataframe
        *X_test: input test dataframe
        *y_train: input target dataframe
        *y_test: input target dataframe
        *today_dt: default as todays date for storing outputs
        *PARAMETER_RANGE: grid search parameters
        *run_tag: tag to identify run (user typed)
        *est: model estimator (options include XGB, LGB, CVB, GB)
        *iterations: model iterations to run
        *seed: seed value
        *vrerbosity: vesrbosity level for output displays (0 for low throughput, 10 default for all)
        *refit_flag: flag to indicate refitting during training
        *cv_split: n-fold input
        *save_pkl: flag to indicate model save (default True)
        stacked: flag to indicate stacking or base models
    '''

    # from sklearn import preprocessing
    # for f in X_train.columns:
    #     if X_train[f].dtype == 'object':
    #         lbl = preprocessing.LabelEncoder()
    #         lbl.fit(list(train[f].values))
    #         X_train[f] = lbl.transform(list(X_train[f].values))
    #
    # for f in X_test.columns:
    #     if X_test[f].dtype == 'object':
    #         lbl = preprocessing.LabelEncoder()
    #         lbl.fit(list(X_test[f].values))
    #         X_test[f] = lbl.transform(list(X_test[f].values))
    #
    # X_train.fillna((-999), inplace=True)
    # X_test.fillna((-999), inplace=True)

    PARAMETER_RANGE = io.get("parameter_range")

    #loading estimators for each model
    estimator = PARAMETER_RANGE[est]['model']
    search_spaces = PARAMETER_RANGE[est]['search_space']
    #estimator = GradientBoostingClassifier()
    
    #setting threads (CB takes a single thread at source)
    if est == 'CB':
        JOB_THREAD = 1
    else:
        JOB_THREAD = 12
        

    #
    # logger.info("Running bayes for {0}\n\n".format(est))

    def status_print(optim_result):
        """Status callback durring bayesian hyperparameter search"""

        # Get all the models tested so far in DataFrame format
        all_models = pd.DataFrame(opt.cv_results_)    

        # Get current parameters and the best parameters    
        best_params = pd.Series(opt.best_params_)
        print('Running for {} \nModel #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
            str(est),
            len(all_models),
            np.round(opt.best_score_, 4),
            opt.best_params_
        ))

    #search_spaces = search_space_model

    opt = BayesSearchCV(estimator, search_spaces, n_iter=ITERATIONS, scoring='roc_auc', 
                        n_jobs=JOB_THREAD, cv=StratifiedKFold(
                                        n_splits=cv_splits,
                                        shuffle=True,
                                        random_state=seed),
                                        refit = refit_flag, 
                                        verbose=verbosity)


    print("*"*300)
    bayes_results = opt.fit(X_train, y_train, callback=status_print)
    logger.info("Bayes run completed for {0}".format(est))
    
    print('\n Best score:')
    print(bayes_results.best_score_)
    #print('\n Best parameters:')
    #print(bayes_results.best_params_)

    means = bayes_results.cv_results_['mean_test_score']
    stds = bayes_results.cv_results_['std_test_score']
    params = bayes_results.cv_results_['params']

    #Saving bayesian runs for each model
    logger.info('Saving bayesian runs for {0} in csv format'.format(est))
    results = pd.DataFrame(bayes_results.cv_results_)

    io.put(f"bayesian_run_{run_tag}_{est}", results)
    logger.info("Fitting using best params for {0}".format(est))
    
    best_params = bayes_results.best_params_
    
    if est == "XGB":
        best_model = XGBClassifier(**best_params)
    elif est == "GB":
        best_model = GradientBoostingClassifier(**best_params)
    elif est == "LGB":
        best_model = LGBMClassifier(**best_params)
    elif est == "CB":
        best_model = CatBoostClassifier(**best_params)
    else:
        logger.error("Wrong model estimator passed in config")

    #Exporting model pickle
    if calibration_method:
        logger.info(f"Start the calibration with {calibration_method} method.")
        best_model = CalibratedClassifierCV(best_model, cv=cv_splits, method=calibration_method)
        # Need to create a separate dataset to train this on TODO

    print("*"*300)
    best_model.fit(X_train, y_train)
    logger.info("Fitted model using best params")



    io.put(f"model_{run_tag}_{est}", best_model)

    #Exporting feature names pickle
    col_list = list(X_train.columns)
    io.put(f"model_features_{run_tag}_{est}", col_list)
    
    
    #best_model = bayes_results.best_estimator_
    logger.info("Predicting customers in test data") 
    prediction_test = best_model.predict(X_test)
    probability_test = best_model.predict_proba(X_test)
    
    #tag for feature importance
    feat_filename = '{0}_{1}'.format(est, run_tag)
    
    #get lift and p/r stats
    feat_imp = evaluate(best_model, X_test, y_test, top_n_features=100, 
                        run_tag = run_tag, today_dt = today_dt,  save_tag = est)
    
    #get percentile wise report
    test_predicted = pd.DataFrame()
    test_predicted['label']=y_test
    test_predicted['prediction']=prediction_test
    test_predicted['prob_1']=probability_test[:,1]
    #print('Sample output from test predicted: ', test_predicted[0:5])
    
    #print conf matrix stats
    confusion_mat(test_predicted, "prob_1", "label", 5, run_tag = run_tag, est=est)

def model_stack_checks(X_train, X_test, y_train, y_test,
                       today_dt, run_tag, stacked, 
                      pickle_path_models = None):
    
    '''
    desc: loads base models from pickles and plots 1st level auc comparison on those
    inputs:
        *today_dt: run date for loading model files
        *run_tag: identifier for model level e.g. layer_1
        *X_train, X_test, y_train, y_test: base model files used for model testing
        *pickle_path_models: pickle where models are saved
        '''
    
    ##reload models
    #run_tag = "layer_1"
    #stacked = False
    try:
        model_xgb =pickle.load(open(pickle_path_models%('XGB', today_dt, run_tag,stacked), 'rb'))
        model_lgb =pickle.load(open(pickle_path_models%('LGB', today_dt, run_tag,stacked), 'rb'))
        #model_gb =pickle.load(open('output/models/xsell_%s_%s_models_smote_%s_stack_%s.pickle'%('GB', today_dt, run_tag,stacked), 'rb'))
        model_cb =pickle.load(open(pickle_path_models%('CB', today_dt, run_tag,stacked), 'rb'))
        logger.info("read all model pickles")
    except:
        logger.warn("all pickles not loaded..pls check filepath again")
    
    
    # Instantiate the classfiers and make a list
    classifiers = [model_xgb, 
                   model_lgb, 
                   #model_gb_cp, 
                   model_cb]

    # Define a result table as a DataFrame
    result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

    # Train the models and record the results
    for cls in classifiers:
        logger.info("sketching for model {0}".format(cls))
        model = cls.fit(X_train, y_train)
        yproba = model.predict_proba(X_test)[::,1]

        fpr, tpr, _ = roc_curve(y_test,  yproba)
        auc = roc_auc_score(y_test, yproba)

        result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                            'fpr':fpr, 
                                            'tpr':tpr, 
                                            'auc':auc}, ignore_index=True)

    # Set name of the classifiers as index labels
    result_table.set_index('classifiers', inplace=True)

    fig = plt.figure(figsize=(8,6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'], 
                 result_table.loc[i]['tpr'], 
                 label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')

    plt.show()
    
    return model_xgb, model_lgb, model_cb

def voting_ensemble(X_train, y_train, X_test, y_test,
                    model_path =  'output/',
                    today_dt ="", est='votingmodel',
                    run_tag = 'life_base',
                    export_path = "",
                    estimators=["XGB"]):
    '''
    desc: loads base models from pickles and runs soft voting ensemble on them
    inputs:
        *model_xgb/model_lgb/model_cb: file paths for model names
        *X_train, X_test, y_train, y_test: base model files used for model testing
        *est: model estimator name
        *run_tag: user declared annotation
        stacked: flag to indicate stacing vs base model
        *out_path: file to store pickles
        '''


    models = []
    #load models from pickle files
    try:
        if "XGB" in estimators:
            model_xgb = io.get(f"model_{run_tag}_XGB")
            logger.info('Read XGB')
            models = models + [('xgb', model_xgb)]

        if "LGB" in estimators:
            model_lgb = io.get(f"model_{run_tag}_LGB")
            logger.info('Read LGB')
            models = models + [('lgb', model_lgb)]

        if "CB" in estimators:
            model_cb = io.get(f"model_{run_tag}_CB")
            logger.info('Read CB')
            models = models + [('cb', model_cb)]
    except:
        logger.critical("Pickles could not be found. Please check file path again")
        raise
    
    
    #building voting classifier
    votingmodel = VotingClassifier(estimators=models, voting='soft')
    
    #fitting voting model
    votingmodel = votingmodel.fit(X_train, y_train)
    logger.info("Fitted model using voting ensemble")
    

    '''
    model pickle export
    '''
    #Exporting model pickle
    io.put(f"model_{run_tag}_{est}", votingmodel)

    #Exporting feature names pickle
    col_list = list(X_train.columns)
    io.put(f"model_features_{run_tag}_{est}", col_list)

    #predicting on test data
    logger.info("Predicting uisng votingmodel on test data") 
    prediction_test_vote = votingmodel.predict(X_test)
    probability_test_vote = votingmodel.predict_proba(X_test)

    #get percentile wise report
    test_predicted = pd.DataFrame()
    test_predicted['label']=y_test
    test_predicted['prediction']=prediction_test_vote
    test_predicted['prob_1']=probability_test_vote[:,1]
    confusion_mat(test_predicted, 'prob_1', 'label',5, run_tag = run_tag, est=est)
    logger.info("Voting ensemble run completed")
    return votingmodel

def evaluate(model_, X_test_, y_test_, top_n_features=200, suffix="",
             run_tag = None, today_dt = None, save_tag = None):
    """
    Evaluate the accuracy, precision and recall of a model
    """
    
    # Get the model predictions
    prediction_test_ = model_.predict(X_test_)
    
    # Print the evaluation metrics as pandas dataframe
    results = pd.DataFrame({"Accuracy" : [metrics.accuracy_score(y_test_, prediction_test_)],
                            "Precision" : [metrics.precision_score(y_test_, prediction_test_)],
                            "Recall" : [metrics.recall_score(y_test_, prediction_test_)]})
    
    # For a more detailed report
    #print(metrics.classification_report(y_test_, prediction_test_))
    print('model in use:', model_)
    #print('textX:', X_test_[0:5])
    #print('textY:', y_test_[0:5])
    
    #computing roc-auc
    fpr, tpr, auc_score = calculate_roc_auc(model_, X_test_, y_test_)
    print("*"*60)
    print("Performance Metrics:")
    print(pd.concat([results, auc_score],axis = 1))
    print("*"*60)
    plot_roc_auc(fpr,tpr)
    # TODO look at how to get  this  for calibrated models
    # feat_imp = pd.DataFrame({'feature': X_test_.columns, 'feature_importance': model_.feature_importances_}).\
    #                     sort_values(by='feature_importance', ascending=False)
    # if suffix !="":
    #     feat_imp['feature'] = feat_imp['feature'].replace({suffix:''}, regex=True)
    # n_row = top_n_features if top_n_features<feat_imp.shape[0] else feat_imp.shape[0]
    # print("*"*60)
    # print("Top {0} important features:".format(n_row))
    # print(feat_imp.head(200))
    #
    # #Save feature importance
    # #Creating directory for model files
    # # out_path_local = '{0}{1}/feature_imp/'.format(export_path, today_dt)
    # # if not os.path.exists(out_path_local):
    # #     os.makedirs(out_path_local)
    #
    # #Saving feature importance for each model
    # #logger.info('Saving important features for {0} in csv format'.format(est))
    #
    # io.put(f"model_feature_importance_{run_tag}_{save_tag}", feat_imp)
    # # logger.info("Feature importance saved to {0}".format(filename_feat))
    #
    #
    # return feat_imp
    return None

# Plot summaries of all charts
def plot_5_pc_charts(testDf, y_test, model = None):
    
    #predict
    prediction_test_ = model.predict(testDf)
    probability_test_ = model.predict_proba(testDf) # For autoGluon not picking index 1 using [:,1]

    #get percentile wise report
    test_predicted = pd.DataFrame()
    test_predicted['label']= y_test
    test_predicted['prediction']=prediction_test_
    test_predicted['prob_1']=probability_test_
    #print('Sample output from test predicted: ', test_predicted[0:5])

    #print conf matrix stats
    confusion_mat(test_predicted, "prob_1", "label", 5, run_tag = 'automl', est='best_model')
    return None
    
def calculate_roc_auc(model_, X_test_, y_test_):
    """
    Evaluate the roc-auc score
    """
    
    # Get the model predictions
    # Note that we are using the prediction for the class 1 -> churn
    #prediction_test_ = model_.predict_proba(X_test_)[:,1]
    
    #removing index selector
    prediction_test_ = model_.predict_proba(X_test_)

    # Compute roc-auc
    fpr, tpr, thresholds = roc_curve(y_test_, prediction_test_)
    
    # Print the evaluation metrics as pandas dataframe
    score = pd.DataFrame({"ROC-AUC" : [metrics.auc(fpr, tpr)]})

    return fpr, tpr, score

def plot_roc_auc(fpr,tpr):
    """
    Plot the Receiver Operating Characteristic from a list
    of true positive rates and false positive rates.
    """
    
    # Initialize plot
    f, ax = plt.subplots(figsize=(7,4))
    
    # Plot ROC
    roc_auc = metrics.auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, alpha=0.3,
                 label="AUC = %0.2f" % (roc_auc))

    # Plot the random line.
    plt.plot([0, 1], [0, 1], linestyle='--', lw=3, color='r',
             label="Random", alpha=.8)
    
    # Fine tune and show the plot.
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC-AUC")
    ax.legend(loc="lower right")
    plt.show()

def pandas_to_list(input_pandas):
    input_pandas_dict = input_pandas.to_dict('index')
    output_list = []
    for key1, value1 in input_pandas_dict.items():
        for key2, value2 in value1.items():
            row_list = [key1, key2, value2]
            output_list.append(row_list)
    return output_list

def to_pickle(para, file_name, para_name = ""):
    """
    desc: Save the parameter to pickle file
    inputs:
        * para: the parameter to be saved to pickle file
        * file_name (str): the pickle file name to be saved to
    returns:
        * None
    """
    with open(file_name, "wb") as picklefile:
        pickle.dump(para, picklefile)
    if para_name != "":
        logger.info("The parameter {0} is saved to the pickle file of {1}".format(para_name, file_name))
    else:
        logger.info("The parameter is saved to the pickle file of {0}".format(file_name))

def load_pickle(file_name, cluster_mode=False, spark=None):
    """
    desc: Load from the saved pickle file
    inputs:
        * file_name (str): the pickle file name, which load the parameter from
    returns:
        * para: the parameter which been saved and now loaded from pickle file
    """
    logger.info("The parameter is loaded from the pickle file of {0}".format(file_name))
    if cluster_mode:
        para = pickle.loads(spark.read.parquet(file_name).collect()[0]["pickle"])
    else:
        with open(file_name, "rb") as picklefile:
            para = pickle.load(picklefile)
    return para


def bays_kfold_optimizer(X, y, fold_constructor,
                         bounds, model, parms_int,
                         splits=5, init_points=10, iterations=15, random_state=1):
    
    fold_params = {
        'shuffle': True
    }
    params_fit = {
        'eval_metric': 'AUC'
    }
    
    def compute_roc_auc(model,index):
        y_predict = model.predict_proba(X.iloc[index])[:,1]
        fpr, tpr, thresh = roc_curve(y.iloc[index], y_predict)
        auc_score = auc(fpr, tpr)
        return fpr, tpr, auc_score
    
    def build_model(**params):
        for param in parms_int:
            params[param] = int(params[param])
        
        #model = model_constructor(**params, **defaults)

        all_auc_val_score = []
        for i in range(folds.n_splits):
                model.fit(X.iloc[X_ids[i],:], y.iloc[X_ids[i]], **params_fit)
                fpr, tpr, auc_score = compute_roc_auc(model, y_ids[i])
                all_auc_val_score.append(auc_score)

        return np.mean(all_auc_val_score)
    
    folds = fold_constructor(n_splits=splits, **fold_params)
    X_ids = []
    y_ids = []

    for (train, test), i in zip(folds.split(X, y), range(splits)):
        X_ids.append(train)
        y_ids.append(test)
        
    model_optimizer = BayesianOptimization(build_model, bounds, random_state=random_state)
    print(model_optimizer.space.keys)

    model_optimizer.maximize(init_points=init_points, n_iter=iterations, acq='ucb', xi=0.0, alpha=1e-6)

    print(model_optimizer.max['target'])
    params = model_optimizer.max['params']
    
    for param in parms_int:
        params[param] = int(params[param])
        
    print(params)
    
    return params

def get_detaildeciles_analysis(df,score,target):
    #  HOW to call: get_detaildeciles_analysis(xgb_df, "score", "target")
    df1 = df[[score,target]].dropna()
    _,bins = pd.qcut(df1[score],100,retbins=True, duplicates = 'drop')
    bins[0] -= 0.0001
    bins[-1] += 0.0001
    bins_labels = ['%d.(%0.2f,%0.2f]'%(99-x[0],x[1][0],x[1][1]) for x in enumerate(zip(bins[:-1],bins[1:]))]
    bins_labels[0] = bins_labels[0].replace('(','[')
    df1['Decile']=pd.cut(df1[score],bins=bins,labels=bins_labels)
    df1['Population']=1
    df1['Zeros']=1-df1[target]
    df1['Ones']=df1[target]
    summary=df1.groupby(['Decile'])[['Ones','Zeros','Population']].sum()
    summary=summary.sort_index(ascending=False)
    summary['TargetRate']=summary['Ones']/summary['Population']
    summary['CumulativeTargetRate']=summary['Ones'].cumsum()/summary['Population'].cumsum()
    summary['TargetsCaptured']=summary['Ones'].cumsum()/summary['Ones'].sum()
    return summary

def generate_feature_direction(input_df, feat_df, include_all_features = False, top_n = 200, group_by = 'prediction',\
                                logging=False, file_dir = ""):
    """
    desc: The function to generate means of each selected features, for groups defined by the input group_by (e.g. predicted take-up vs non take-up), thus it can be used to infer the feature directions.
    inputs:
        * input_df: the input spark dataframe with all the values for each feature to be profiled, e.g. the prediction spark dataframe.
        * feat_df: the pandas feature dataframe which stores the feature importance list, sort by feature importance score (in descending order).
        * include_all_features (Boolean, optional): the flag to indicate whether include all features for calculation.
        * top_n (int, optional): the top n features which is calculated for the feature direction. It will take effect when  include_all_features = False.
        * group_by (str, optional): decide how to group the base. e.g. by the column of "prediction" (comparing predicted take-up vs non take-up) or "label" (comparing actual take-up vs non take-up).
        * logging (Boolean, optional): flag to indicate whether save the resulted feature mean.
        * file_dir (str, optional): the path which the resulted feature mean is saved to if logging = True.
    returns:
        * means: the pandas dataframe which store the means of each selected features, for groups defined by the input group_by (e.g. predicted take-up vs non take-up), thus it can be used to infer the feature directions.
    """
    '''
        desc: the feature directions is decided by based on model prediction
        include_all_features will only include all features in feat_df, which are selected features
    '''
    if include_all_features:
        top_features = list(feat_df['feature']) + [group_by]
        print("All the features will be measured.")
    else:
        # if the number of selected features less than top_n, it will automatically use the all the features.
        top_features = list(feat_df[:top_n]['feature']) + [group_by]
        print("Top {0} features will be measured.".format(len(top_features)-1))

#     # cast the decimal to float to speed up the pandas conversion
#     input_df = util.spark_df_decimal_cast(input_df)
#     pd_predictions = input_df.select(top_features).toPandas()
    
    pd_predictions = input_df[top_features]

    means = pd.DataFrame()
    for k in top_features:
        means.loc[k,'Postive mean'] = pd_predictions.loc[pd_predictions[group_by] == 1][k].mean()
        means.loc[k,'Negative mean'] = pd_predictions.loc[pd_predictions[group_by] == 0][k].mean()
    means = means.reset_index()
    means.rename({'index':'Feature'}, axis=1, inplace=True)
    means['Difference'] = means['Postive mean']-means['Negative mean']
    print("For Top 10 features:")
    print(means[:10])
    if logging:
        means.to_csv(file_dir)
        print("The output feature mean is saved to {0}".format(file_dir))
    return means

def __print_hit_cov(matrix, base_rate = None, print_flag=True):
    """
    desc: The internal function to print the hit rate and coverage from the confusion matrix.
    inputs:
        * matrix: the confusion matrix.
    returns:
        * None
    """
    if matrix[0,1] + matrix[1,1] != 0:
        hit_rate = float(matrix[1,1])/ (matrix[0,1] + matrix[1,1])
        if print_flag:
            print('------------------ Hit rate / Precision: {0:.3%}'.format(hit_rate))
        if base_rate != None:
            lift = hit_rate/base_rate
            if print_flag:
                print('---------------------------------- Lift: {0:.3}'.format(lift))
    else:
        hit_rate = -1
        lift = -1
        if print_flag:
            print('------------------ Hit rate / Precision: N.A.')
    if matrix[1,0] + matrix[1,1] != 0:
        coverage = float(matrix[1,1])/ (matrix[1,0] + matrix[1,1])
        if print_flag:
            print('--------------------- Coverage / Recall: {0:.3%}'.format(coverage))
    else:
        coverage = -1
        if print_flag:
            print('--------------------- Coverage / Recall: N.A.')
    return hit_rate, coverage, lift

def __print_confusion_matrix(cm):
    '''
    generates a printable format of confusion matrix
    '''
    cm_list = cm.tolist()
    #cm_list = cm
    cm_list[0].insert(0, '0')
    cm_list[1].insert(0, '1')
    print(tabulate.tabulate(cm_list, headers=['Actual/Pred', '0', '1']))

def __plot_lift_chart(lift_list):
    '''
    takes lift lift and plots it inline
    '''
    plt.plot(range(1,11,1),lift_list)
    plt.xticks(range(1,11,1))
    plt.grid(True)
    plt.title('Lift Chart', fontsize=15)
    plt.xlabel("Decile",fontsize=12)
    plt.ylabel("Lift",fontsize=12)
    plt.show()

def confusion_mat(input_pandas_df, pos_prob_colname, model_label, top_n_decile, run_tag = "", est="", overall_mat=True, prediction="prediction", print_top_deciles = False, return_lift=False, return_takeup=False):
    """
    desc: The function to output the confusion matrix with hit rate and coverage.
    inputs:
        * input_pandas_df: the pandas dataframe, which has 4 columns: key_field, prediction, model_label, output_pos_prob. It is the output from the function of model_prediction in modeling.py.
        * pos_prob_colname (str): the output column name for the probability (the probability for y=1).
        * model_label (str): the label column for the model training.
        * top_n_decile (int): until which decile the confusion matrix is print.
        * overall_mat (Boolean, optional): the flag to indicate whether to print overall confusion matrix, which uses 0.5 for probability threshold.
        * prediction (str, optional): the column name for the model prediction.
        * print_top_deciles (Boolean, optional): the flag to indicate whether to print last few lines of top deciles dataframe. It is for the purpose of checking probability for each decile.
    returns:
        * None
    """

    '''
        prediction: the column name for prediction which is used for overall confusion matrix
    '''
    df = input_pandas_df.copy()
    df = df.sort_values(pos_prob_colname, ascending=False)
    base_rate = float(sum(df[model_label]))/len(df[model_label])
    print('Real base rate: {0:.3%}'.format(base_rate))
    print("Top few lines of the prediction dataframe: ")
    print(df[:5])
#     print '----'
#     print df[-5:]

#     update the decile dfs:
#     top_5_percent = df[:len(df)/20*1]
#     top_1_deciles = df[:len(df)/10*1]
#     top_2_deciles = df[:len(df)/10*2]
#     top_3_deciles = df[:len(df)/10*3]
#     top_4_deciles = df[:len(df)/10*4]
#     top_6_deciles = df[:len(df)/10*6]
    if overall_mat:
        conf_matrix = confusion_matrix(df.loc[:, model_label], df.loc[:, prediction]) # just use 0.5 threshold for overall confusion metrics
        print('----'*15)
        #print(conf_matrix)
        print('Overall Confusion Matrix: ')
        __print_confusion_matrix(conf_matrix)
        # print the hit rate and coverage
        hit_rate, coverage, lift = __print_hit_cov(conf_matrix, base_rate=base_rate)

    prediction = "prediction_decile"

    df['row'] = df.reset_index().index+1
    
    # always show the top 1 percent
    top_1_percent = df.copy()
    top_1_percent[prediction] = np.where(top_1_percent['row']<=(len(df)/100*1), 1, 0)
    top_1_percent_conf_matrix = confusion_matrix(top_1_percent[model_label], top_1_percent[prediction])
    #     print '----'*4
    print('Top 1 percent')
    __print_confusion_matrix(top_1_percent_conf_matrix)
    hit_rate, coverage, lift = __print_hit_cov(top_1_percent_conf_matrix, base_rate=base_rate)

    # always show the top 5 percent
    top_5_percent = df.copy()
    top_5_percent[prediction] = np.where(top_5_percent['row']<=(len(df)/20*1), 1, 0)
    top_5_percent_conf_matrix = confusion_matrix(top_5_percent[model_label], top_5_percent[prediction])
    #     print '----'*4
    print('Top 5 percent')
    __print_confusion_matrix(top_5_percent_conf_matrix)
    hit_rate, coverage, lift = __print_hit_cov(top_5_percent_conf_matrix, base_rate=base_rate)
    
    lift_list=[]
    takeup_list=[]

    for i in range(10):
        i = i+1
        top_decile = df.copy()
        top_decile[prediction] = np.where(top_decile['row']<=(len(top_decile)/10*i), 1, 0)
        top_decile_conf_matrix = confusion_matrix(top_decile[model_label], top_decile[prediction])
        
        if i<= top_n_decile:
            print('####'*15)
            print('------------------------ The {0} decile ---------------------'.format(i))
            if print_top_deciles:
                top_deciles_data = df[:len(df)/10*i]
                print("The last few lines of top deciles dataframe: ")
                print(top_deciles_data[-5:])
            __print_confusion_matrix(top_decile_conf_matrix)
            # print the hit rate and coverage for each decile
            hit_rate, coverage, lift = __print_hit_cov(top_decile_conf_matrix, base_rate=base_rate)
        else:
            hit_rate, coverage, lift = __print_hit_cov(top_decile_conf_matrix, base_rate=base_rate,print_flag=False)
        lift_list.append(lift)
        takeup_list.append(top_decile_conf_matrix[1,1])
    
    #__plot_lift_chart(lift_list)
    __get_lift_precision_recall(lift_list, input_pandas_df, pos_prob_colname, model_label, run_tag=run_tag, est=est)
    
#    if return_lift and not return_takeup:
#        return lift_list
    
#    if return_takeup:
#        return lift_list, takeup_list 


def __get_lift_precision_recall(lift_list, df,score,target,run_tag ="", est="", title=""):
        
    plt.figure(1,figsize=(20, 5))
    
    plt.subplot(121)
    plt.plot(range(1,11,1),lift_list)
#     print(lift_list)
    plt.xticks(range(1,11,1))
    plt.grid(True)
    plt.title('Lift Chart', fontsize=15)
    plt.xlabel("Decile",fontsize=15)
    plt.ylabel("Lift",fontsize=15)
    
    plt.subplot(122)
    df1 = df[[score,target]].dropna()
    fpr, tpr, thresholds = roc_curve(df1[target], df1[score])
    ppr=(tpr*df[target].sum()+fpr*(df[target].count()-df[target].sum()))/df[target].count()
    gain_df = pd.DataFrame()
    gain_df['x']=ppr
    gain_df['y']=tpr
#     print(gain_df)
#     plt.figure(figsize=(12,4))
    #plt.subplot(1,2,1)
    plt.plot(ppr, tpr, label='')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(b=True)
    plt.xlabel('%Population', fontsize=15)
    plt.ylabel('%Target', fontsize=15)
    plt.title(title+'Cumulative Gains Chart', fontsize=15)
    plt.legend(('Choosen model', 'Default'), frameon=True, loc="lower right")
    
    #save plots
    # savefile = '{0}lift_gain_chart.png'.format(export_path)
    io.put(f"lift_gain_chart_{run_tag}_{est}", plt)

    
    #display
    plt.tight_layout()    
    plt.show()
    

def holdout_stats(holdoutdf, idcols = '', primary_target = 'target_flag',
                 import_path = 'output/',
                 today_dt ='',
                 run_tag = '',
                 reference_percentile = 3,
                  estimators=['XGB']):

    holdout_train=holdoutdf.drop(idcols, axis=1)
    holdout_target=holdoutdf[primary_target]
    logger.info('distribution of target in holdout data \n**************\n{0}'\
               .format(holdoutdf[primary_target].value_counts()))

    #Creating directory for model files
    # out_path_local = '{0}{1}/models/'.format(import_path, today_dt)

    #load models from pickle files
    models = []
    try:
        if "XGB" in estimators:
            model_xgb = io.get(f"model_{run_tag}_XGB")
            models = models + [("XGB", model_xgb)]
            logger.info('Read XGB')

        if "LGB" in estimators:
            model_lgb = io.get(f"model_{run_tag}_LGB")
            models = models + [("LGB", model_lgb)]
            logger.info('Read LGB')

        if "CB" in estimators:
            model_cb = io.get(f"model_{run_tag}_CB")
            models = models + [("CB", model_cb)]
            logger.info('Read CB')

        model_voting = io.get(f"model_{run_tag}_votingmodel")
        models = models + [("votingmodel", model_voting)]
        logger.info('Read voting model')
    except:
        logger.critical("Pickles could not be loaded. Please check file path again")
        raise

    '''
    triggering all models
    '''
    #preparing dataframe to hold results
    temp_df = pd.DataFrame(columns=['top_n','hitrate_prec', 'coverage_rec', 'lift'])
    compare_df = pd.DataFrame()

    
    #triggering model performance loops
    out_img = '{0}{1}/holdout_performance/'.format(import_path, today_dt)
    if not os.path.exists(out_img):
        os.makedirs(out_img)

    for est_name, est in models:
        logger.info('Running predictions on holdout for {0}'.format(est))
        prediction_test_holdout = est.predict(holdout_train)
        probability_test_holdout = est.predict_proba(holdout_train)

        #get percentile wise report
        test_predicted_ho = pd.DataFrame()
        test_predicted_ho['label']=holdout_target
        test_predicted_ho['prediction']=prediction_test_holdout
        test_predicted_ho['prob_1']=probability_test_holdout[:,1]
        
        #filename to save images (lift and gain charts) for models on holdout
        file_suffix = str(est)[0:10]
        imgout = '{0}model_{1}_holdout_'.format(out_img,file_suffix)
        logger.info('saving model performance image for {0}'.format(file_suffix))
        confusion_mat(test_predicted_ho, 'prob_1', 'label',5, run_tag=run_tag, est=f"{est_name}_holdout")

        for val in np.arange(1,6):
            hr, cov, lift = lift_top(test_predicted_ho, 'prob_1', 'prediction', 'label', val)
            #print(hr)
            #print(cov)
            #print(lift)
            temp_df = temp_df.append({'model_name': est, 'top_n': val, 'hitrate_prec': hr, 'coverage_rec': cov,
                                      'lift': lift}, 
                         ignore_index=True)
            #temp_df['hitrate_prec'] = hr
            #print(temp_df)
            #temp_df['coverage_rec'] = cov
            #temp_df['lift'] = lift
            compare_df = pd.concat([compare_df, temp_df])
            #print(temp_df)
            
    #reference table
    reference_df = temp_df.loc[temp_df.top_n == reference_percentile]
    best_model = reference_df.loc[(reference_df['lift'].idxmax())].model_name
    
    #export temp file
    io.put(f"all_holdout_comparison_out_{run_tag}", temp_df)
    return temp_df, best_model

def lift_top(input_pandas_df, pred_values, pred_label, true_label, pcnt):
    df = input_pandas_df.copy()
    df = df.sort_values(pred_values, ascending=False)
    df['row'] = df.reset_index().index+1
    base_rate = float(sum(df[true_label]))/len(df[true_label])
    print('base rate = ', base_rate*100)
    top_n_percent = df.copy()
    #top_n_percent = top_n_percent.sort_values(pred_values, ascending=False)
    #top n percent
    top_n_percent[pred_label] = np.where(top_n_percent['row']<=(len(top_n_percent)/(100/pcnt)*1), 1, 0)
    top_n_percent_conf_matrix = confusion_matrix(top_n_percent[true_label], top_n_percent[pred_label])
    print(confusion_matrix)
    #     print '----'*4
    print('Top {0} percent'.format(pcnt))
    __print_confusion_matrix(top_n_percent_conf_matrix)
    hit_rate, coverage, lift = __print_hit_cov(top_n_percent_conf_matrix, 
                                               base_rate=base_rate) 
    return hit_rate, coverage, lift

import pandas as pd
import numpy as np
import pickle
import gc
import os
from collections import Counter
import json
from scipy import stats
import csv,argparse
import tabulate

#from util.io_layer import io_ds as io
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from skopt.space import Real, Categorical, Integer
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from bayes_opt import BayesianOptimization
from catboost import CatBoostClassifier

#from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.metrics import auc, roc_auc_score, roc_curve, f1_score, mean_squared_error, confusion_matrix,make_scorer, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFE
from skopt import BayesSearchCV
from bayes_opt import BayesianOptimization

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import datetime
from dateutil.relativedelta import relativedelta

from logger import logger

from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

#custom
#import models.model_training as mt

def score_leads(df, best_model = None, idcols = '', primary_key = 'customerseqid',\
                primary_target = 'target_flag',\
                export_path_local = 'output/', run_tag = None, today_dt = None):
    '''
    desc: scores the leads using the model provided as input and exports the leads to a designated s3 path
    inputs
        *df: dataframe to be scored
        *best_model: model file to be scored with
        *idcols: list of columns that are identifiers and need to be excluded from scoring
        *primary_key
        *primary_target
        *export_path_local: local folder to export file
        *export_path_s3: path to export leads file to s3
        today_dt: variable instantiated with run date of the model
    '''
    #df = dataframe.copy()
    score_train=df.drop(idcols, axis=1)
    score_target=df[primary_target]
    logger.info('Distribution of target in holdout data \n**************\n{0}'\
           .format(df[primary_target].value_counts()))
    
    #creating folder to dump all files
    # out_path = '{0}{1}/scored/'.format(export_path_local, today_dt)
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)
        
    logger.info('Scoring data using {0}'.format(str(best_model)))
    prediction_scored = best_model.predict(score_train)
    probability_scored = best_model.predict_proba(score_train)

    #get percentile wise report
    out_preds = pd.DataFrame()
    out_preds['label']=score_target
    out_preds['prediction']=prediction_scored
    out_preds['prob_1']=probability_scored[:,1]
    
    #report
    mt.confusion_mat(out_preds, 'prob_1', 'label',5, run_tag=run_tag, est="scored")
    
    #export leads out
    dfout = df.copy()
    dfout['propensity'] = probability_scored[:,1]

    ##final scored data
    #print(dfout.propensity.head())
    #print(dfout[primary_key].head())
    dfout = dfout.loc[:, [primary_key, 'propensity']]\
                .sort_values('propensity', ascending = False)
    
    ### ranking and adding percentile
    dfout['percentile'] = 100 - pd.qcut(dfout.propensity, 100, labels=False, duplicates='drop')
    #export to s3
    io.put(f"leads_{run_tag}", dfout)
    logger.info('Leads scoring completed and file has been pushed to S3')
    logger.info("Final scored data stats:\n****************\nTotal rows = {0} and total cols = {1}"\
            .format(dfout.shape[0], dfout.shape[1]))
    
    return dfout

def split_into_l_g(leads_df, base_df, out_path_g = None, out_path_l = None, score_month = 202007, today_dt = None):
    #import datetimessss
    #today_dt = datetime.datetime.today().strftime('%Y-%m-%d')
    #add cols to match
    keepcols = ['customerseqid', 'prod_held_g_only', 'prod_held_l_only', 'prod_held_l_and_g']
    
    #filter basedf for specific month
    base_df = base_df.loc[base_df.monthyearid == score_month]
    leads_df = leads_df.merge(base_df.loc[:, keepcols], on = ['customerseqid'])
    
    #filter for l/g leads
    g_leads = leads_df.loc[leads_df.prod_held_g_only == 1]\
            .drop(['prod_held_g_only', 'prod_held_l_only', 'prod_held_l_and_g'], axis = 1)
    l_leads = leads_df.loc[(leads_df.prod_held_l_and_g == 1) | (leads_df.prod_held_l_only == 1)]\
            .drop(['prod_held_g_only', 'prod_held_l_only', 'prod_held_l_and_g'], axis = 1)
    
    #push to s3
    l_leads.to_csv(out_path_l.format(today_dt))
    g_leads.to_csv(out_path_g.format(today_dt))
   
    logger.info('L and G x-sell health files pushed to s3')
    return l_leads, g_leads    

from itertools import product
import pandas as pd
import numpy as np

# 1. Building CA
def building_CA(df):
    """
    1 - If leads is attached, keep only (c,a) where a is agent
    2 - If leads is underserved, keep only (c,a) where a is opt in underserved
    3 - If leads is orphan, keep only agents that are orphan opt in orphans
    4 - If customer language is known, select only agents with same language
    """
    
    #attached + underserved
    #------------------
    attached_CA = df[df['customertype'].isin(['notorphan', 'partialorphan'])] \
                       [['customerseqid', 'agentseqid']].drop_duplicates()
    attached_CA = list(attached_CA.to_records(index=False))
    
    #orphans
    #------------------
    agent_for_orphans = list(df[df['orphanoptinflag']=='Y']['agentseqid'].unique())
    orphans = list(df[df['customertype']=='fullyorphan']['customerseqid'].unique())
    orphan_CA = list(product(orphans,agent_for_orphans))
    
    #placeholder for language
    # TODO - Carlos to help
        
    return attached_CA + orphan_CA    



# 2. CAP
def building_CAP(df, CA):
    """
    1 - cartesian product CA x CP
    2 - filter out comibnaisons of c,a,p based on rules for HNW opt out (orphan and attached)
    # TODO - what if an agent has 0 attached in the exclusions ?
    """

    CAP = pd.DataFrame([[x[0],x[1]] for x in CA],columns=['customerseqid','agentseqid'])\
            .merge(df[['customerseqid','usecase']].drop_duplicates(),on=['customerseqid'],how='inner')
    
    CAP = list(CAP.to_records(index=False))
    print('CAP:',len(CAP),'size of problem')
    
    return CAP

# 3. Cw
def build_CWarm(df,CAP):
    '''
    - select only underserved and orphans
    '''        
    Cwarm = list(df[(df['customertype'].isin(['fullyorphan'])) | (df['underservedflag']=='Y')]\
        ['customerseqid'].unique())
    
    # Reduce CwAP now
    CwAP = pd.DataFrame([[ x[0],x[1],x[2]] for x in CAP],columns=['c','a','p'] )
    CwAP = list(CwAP[CwAP['c'].isin(Cwarm)].to_records(index=False))
    
    return Cwarm,CwAP


# 4. other domains
def building_other_domain(df, CAP):
    '''
    multiplication or drop duplicates of th right ones
    add all in domain
    '''
    # main domains
    C = list(set([x[0] for x in CAP]))
    A = list(set([x[1] for x in CAP]))
    P = list(set([x[2] for x in CAP]))
    
    # Build AP / CP
    AP = list(set([(x[1],x[2]) for x in CAP]) )
    CP = list(set([(x[0],x[2]) for x in CAP]) )
    
    #Build AT
    AT  = df[(df['channel']!='orphan')][['agentseqid', 'channel']].drop_duplicates()
    AT = list(AT.to_records(index=False))
    
    return C,A,P,AP,CP,AT



#############################################################
# main
#############################################################


def build_domains(df):
    
    import time
    start = time.time()

    d = {}
    # computing CA
    d['CA'] = building_CA(df)
    print('CA',np.round(time.time() - start,0))

    # compiuting CAP
    d['CAP'] = building_CAP(df, d['CA'])
    print('CAP',np.round(time.time() - start,0))

    # computing Cw
    d['Cw'],d['CwAP'] = build_CWarm(df, d['CAP'])
    print('CwAP',np.round(time.time() - start,0))

    # computing other domains
    d['C'],d['A'] ,d['P'] ,d['AP'] ,d['CP'] ,d['AT']  = building_other_domain(df, d['CAP'])
    print('other',np.round(time.time() - start,0))

    return d

#native libraries
import pandas as pd
import numpy as np
import os
from os import path
import sys
import warnings
import logging
import configparser
warnings.filterwarnings("ignore")

#sub utils
import datetime

#opti functions (google or tools)

#from __future__ import print_function
from ortools.linear_solver import pywraplp


# display formatter
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)

# setup custom logger
logger = logging.getLogger(__name__)


######################
# Allocation
#######################

# 1. Min max capacity per agent
def minMaxCapaAgt(AP,A):
    '''
    Min leads per agent (todo: pass as dict)
    ------
    # TODO - replace with actual value from files for max / confing
    # TODO - replace with actual value from files for min / read from file alyssa
    '''

    # 0 to be changed
    capa_min = {(a,p):0 for (a,p) in AP}

    # random to be changed
    rand = np.random.randint(100,120,len(A))
    capa_max = {A[i] :rand[i] for i in range(len(A))}
    
    return capa_min,capa_max


# expected value definition
def compute_expected_values(df,
                            CP,
                            info =  {"fpp": [0.08,4000],
                                   'maturity':[0.2,4200],
                                   'hnw':[0.05,6000],
                                   'upsell_life_hp':[0.02,5200],
                                   'xsell_health_gi':[0.001,700],
                                    'upsell_life_rp':[0.002,6100],
                                   'xsell_health_life':[0.001,1300],
                                   'xsell_life_hi': [0.001,6500],
                                  }
                                ):
    '''
    Compute expected value for customers
    TODO - move to config
    TODO (not for MVP) - per decile to have a "real" conversion rate
    ------
    Simplified logic for now: (1 - (expectedValue.percentile/20))*expectedValue.wp *expectedValue.cr
    '''
    percentile = df.set_index(['customerseqid','usecase'])['percentile'].to_dict()
    
    expectedValue = { (c,p): (1 -percentile[(c,p)] / 20) *info[p][1] * info[p][0] * 1000 for (c,p) in CP}
    
    return expectedValue

"""
def compute_affinity(CA):
    '''
    Compute distance * weight of each (c,a) if c orphan
    if c non orphan assign 0 - bigger the better
    # TODO - add affinity
    -------
    Simplified logic: Random
    '''
    rand = np.random.uniform(0,1,len(CA))
    affinity = {  (CA[i][0],CA[i][1]) : rand[i] for i in range(len(CA))}

    return affinity
"""

# !!!!!!!!!!!
## Remove after paul integrates


## affinity between customer and agent

def affinity_est(df, cust_master, agent_master):
    '''
    Returns affinity between customer <> agent in CA dataframe. higher the better
    Inputs:
        df: master leads dataframe
        cust_master:Customer level metadata
        agent_master: Agent level metadata
    '''
    
    #build CA
    #attached + underserved
    #------------------
    attached_CA = df[df['customertype'].isin(['notorphan', 'partialorphan'])] \
                       [['customerseqid', 'agentseqid']].drop_duplicates()
    
    #orphans
    #------------------
    agent_for_orphans = df[df['orphanoptinflag']=='Y'][['agentseqid']].drop_duplicates()
    orphans = df[df['customertype']=='fullyorphan'][['customerseqid']].drop_duplicates()
    
    #cross join for orphans
    agent_for_orphans['tmp'] = 1
    orphans['tmp'] = 1
    orphan_CA = orphans.merge(agent_for_orphans, on = 'tmp', how = 'outer').drop('tmp',axis=1)
    
    CA = pd.concat([attached_CA, orphan_CA])
    
    ## calculate hh serving score
    #merge leads and customer
    _tmp = CA.merge(cust_master, on = 'customerseqid', how = 'left')
    
    #rename agentseqid for checking hh_service status
    agent_master = agent_master.rename({'agentseqid' : 'agentseqid1'}, axis = 1)
    
    #merge agent info    
    _tmp = _tmp.merge(agent_master, left_on = ['agentseqid', 'customer_hid'], \
                      right_on = ['agentseqid1', 'agent_hid'], how = 'left')
    
    _tmp['hh_service'] = np.where(_tmp.agentseqid1.isnull() | _tmp.agent_hid.isnull() ,0,1)
    
    #pick max
    _tmp_agg = _tmp.groupby(['customerseqid', 'agentseqid']) \
        .agg({"hh_service" : max}).reset_index()
    
    ## merge CA and scores
    
    #merge CA and warming info
    ##add hh serving info (if agent serves hh then 1 else 0)
    cust_agent = CA.merge(_tmp_agg, on = ['customerseqid', 'agentseqid'],\
                                how = 'left')
    
    #add agent info
    select_agent_cols = ['agentseqid1', 'agent_language', 'agent_postal_sector', 'agent_age']
    cust_agent = cust_agent.merge(agent_master.loc[:,select_agent_cols].drop_duplicates()\
                                  , left_on = ['agentseqid'], right_on = ['agentseqid1'], how ='left')
    
    #add customer info
    select_cust_cols = ['customerseqid', 'customer_language', 'customer_postal_sector', 'customer_age']
    cust_agent = cust_agent.merge(cust_master.loc[:,select_cust_cols].drop_duplicates()\
                                  , on = ['customerseqid'], how = 'left')
    
    ##add scores using the logic as - 
    # same language  = 1, same postal sector =1, age diff between less than 5 years = 1 and if agent serves HH then 1
    
    cust_agent['lang_match'] = np.where(((cust_agent.customer_language.isnull()) | cust_agent.agent_language.isnull() | (cust_agent.customer_language != cust_agent.agent_language)), 0, 1)
    cust_agent['postal_match'] = np.where(((cust_agent.customer_postal_sector.isnull()) | (cust_agent.agent_postal_sector.isnull()) | (cust_agent.customer_postal_sector != cust_agent.agent_postal_sector)), 0, 1)
    cust_agent['age_match'] = np.where((cust_agent.customer_age == cust_agent.agent_age).abs() <=5,1,0)
    
    #return final scores (linear sum)    
    cust_agent['affinity_score'] = (cust_agent.lang_match + cust_agent.postal_match + cust_agent.age_match \
            + cust_agent.hh_service) * 100
    
    #cols retain
    cols_retain = ['customerseqid', 'agentseqid', 'affinity_score']
    df= cust_agent.loc[:, cols_retain]
    
    #return
    
    affinity_score = df.set_index(['customerseqid','agentseqid'])['affinity_score'].to_dict()
    
    affinity = { (c,a): affinity_score[(c,a)] for (c,a) in df.loc[:, ['customerseqid','agentseqid']].values}
    
    return affinity


def compute_agentPriority(df,CA):
    '''
    if orphan - assign 0
   if non orphan priority rules to compute - bigger the better
    # TODO - replace 1 for priority_not_orphan (not P1)
    ----
    Simplified logic :0 if orphan, random .5,1 if not
    '''
    # orphans
    orphan_dic = df[['customerseqid','customertype']].drop_duplicates('customerseqid')\
                                .set_index(['customerseqid'])['customertype'].to_dict()
    
    CA_o =  [ x for x in CA if orphan_dic[x[0]] == 'fullyorphan']
    priority_orphan = {(c,a): 0 for (c,a) in CA_o }
    
    # not orpahn - random
    CA_no = [ x for x in CA if orphan_dic[x[0]] != 'fullyorphan']
    priority_not_orphan = {  (CA_no[i][0],CA_no[i][1]) : 1 for i in range(len(CA_no))}
    
    # merge
    priority  = {**priority_orphan, **priority_not_orphan}

    return priority  

def build_duo_domain(CAP):
    # recompute dataframe
    df = pd.DataFrame([[x[0],x[1],x[2]] for x in CAP],columns=['c','a','p'])
    
    # iterate on chunks
    APdomain = {}
    for c,chunk in df.groupby('c'):
        APdomain[c] = list(chunk[['a','p']].to_records(index=False))
    
    CAdomain = {}
    for p,chunk in df.groupby('p'):
        CAdomain[p] = list(chunk[['c','a']].to_records(index=False))

    Cdomain = {}
    for ap,chunk in df.groupby(['a','p']):
        Cdomain[ap] = list(chunk['c'].unique())

    return APdomain,CAdomain,Cdomain


def product_min_max(P, info =  {"fpp": [0,4000],
                            'maturity':[0,4000],
                            'hnw':[0,4000],
                            'upsell_life_hp':[0,4000],
                            'xsell_health_gi':[0,4000],
                            'upsell_life_rp':[0,4000],
                            'xsell_health_life':[0,4000],
                            'xsell_life_hi': [0,4000],
                                  }):
    """
    # TODO - to take it out into config
    """
    
    min_max_product = {p:{'min':info[p][0], "max":info[p][1] } for p in P}
    return min_max_product


def build_CPattached_CPoprhan(df,CAP):
    """
    filter each CP based on attached or not
    """
    def create_dic(CAP_reduced):
        dic = {}
        df = pd.DataFrame([[x[0],x[1],x[2]] for x in CAP_reduced],columns=['c','a','p'])
        # iterate on chunks
        for a,chunk in df.groupby('a'):
                dic[a] = list(chunk[['c','p']].to_records(index=False))
        return dic      

    # orphan dictionnary
    orphan_dic = df[['customerseqid','customertype']].drop_duplicates('customerseqid')\
                                .set_index(['customerseqid'])['customertype'].to_dict()  
    
    
    # difference between each list
    CAP_o =  [ x for x in CAP if orphan_dic[x[0]] == 'fullyorphan']
    CPorphan = create_dic(CAP_o)
    CAP_no =  [ x for x in CAP if orphan_dic[x[0]] != 'fullyorphan']
    CPattached = create_dic(CAP_no)
    
    return CPattached,CPorphan

######################
# Warming
#######################

def build_Awarming(A):
    """
    TODO - to change to read a file
    Anowarming - Agents who are warming opt out
    Awarming_compuls - Agents who are compilsory warming
    """
    
    rand = np.random.uniform(0,1,len(A))
    agents = {  A[i]: rand[i] for i in range(len(A))}

    Anowarming = [a for a in A if agents[a] <= 0.2]
    Awarming_compuls = [a for a in A if agents[a] >= 0.8]

    return Anowarming,Awarming_compuls


def build_warmingScore(CwAP):
    '''
    rules to be applied - bigger the better
    --------
    # TODO - compute real score
    ''' 
    rand =  np.random.randint(100,200,len(CwAP))
    warmingScore = {  (CwAP[i][0],CwAP[i][1],CwAP[i][2]): rand[i] for i in range(len(CwAP))}

    return warmingScore

##build warming score
# !!!!!!!!!!!!!!!!!!!!!

def warming_score_calc(CwAP, master_df, p_priority):
    '''
    Returns incremental warming weights for CAP matrix. Higher the better
    Inputs:
        CAP: CAP dataframe
        master_df: Master leads data from s3 (leadsOut)
        p_priority: dict containing use case level priorities (1 is best)
        cust_info:Customer level metadata
        agent_info: Agent level metadata
    '''
    
    
    #select relevant columns from leads master
    _tmp = master_df.loc[:, ['customerseqid', 'agentseqid', 'days_underserved', 'wpi']].copy()
    
    #impute wpi and days_underserved
    #TODO: check if its required
    _tmp['wpi'] = _tmp['wpi'].fillna(0)
    _tmp['days_underserved'] = _tmp['days_underserved'].fillna(9999)
    
    #rank wpi and days_underserved
    _tmp['wpi_score'] = pd.qcut(_tmp.wpi, 10, duplicates = 'drop',labels= False)
    _tmp['serving_score'] = pd.qcut(_tmp.days_underserved, 10, duplicates = 'drop', labels= False)
    
    #add CwAP params
    
    CwAP = pd.DataFrame([[ x[0],x[1],x[2]] for x in CwAP],columns=['customerseqid','agentseqid','usecase'] )
    
    
    CwAP = CwAP.merge(_tmp, on = ['customerseqid', 'agentseqid'], how = 'left')
    
    CwAP = CwAP.merge(p_priority.loc[:, ['usecase', 'priority']], on = 'usecase')
    
    #impute_missing_values
    CwAP['wpi_score'] = CwAP['wpi_score'].fillna(0)
    CwAP['serving_score'] = CwAP['serving_score'].fillna(9999)
    CwAP = CwAP.drop_duplicates()
    
    #scoring 3 params
    CwAP['warming_score'] = CwAP['wpi_score'].astype(int) * CwAP['serving_score'].astype(int) \
                        * (10- CwAP['priority']).astype(int)
    
    warmingScore_in = CwAP.set_index(['customerseqid','agentseqid','usecase'])['warming_score'].to_dict()
    
    warmingScore = { (c,a,p): warmingScore_in[(c,a,p)] for (c,a,p) in CwAP.loc[:, ['customerseqid','agentseqid','usecase']].values}
    
    return warmingScore


def max_warming_type():
    """
    # TODO - change with value from file alyssa / config ?
    """
    warming_capa = {'rfs': 10000, 'agency': 5000}
    return warming_capa

def build_warming_domains(CwAP,AT):
    """
    Return the domain of the function
    """
    CwAPdomain = {}
    for t in ['rfs','agency']:
        agents = [x[0] for x in AT if x[1] == t]
        CwAPdomain [t] = [x for x in CwAP if x[1] in agents]
    
    return CwAPdomain

def build_CwPdomain(CwAP):
    """
    CwPdomain is per agent all the CwP
    """
    dic = {}
    
    df = pd.DataFrame([[x[0],x[1],x[2]] for x in CwAP],columns=['c','a','p'])
    # iterate on chunks
    for a,chunk in df.groupby('a'):
        dic[a] = list(chunk[['c','p']].to_records(index=False))

    return dic


#############################################################
# main
#############################################################

                         ##building use case conversion matrix
cr_dict = {'usecase': ['fpp', 'maturity', 'hnw', 'upsell_life_hp', 'xsell_health_gi',
       'upsell_life_rp', 'xsell_health_life', 'xsell_life_hi'], 
           'cr': [.08, .2, .05, .05, .001,
       .002, .001, .001],
          'wp': [4000, 4200, 6000, 5200, 700,
       6100, 1300, 6500],
          'min': [10, 10, 10, 10, 10,
       10, 10, 10],
          'max': [1000, 1000, 4000, 4000, 4000,
       4000, 4000, 4000],
          'priority': [1, 2, 3, 5, 7,
       4, 6, 8]}

cr_df = pd.DataFrame(data=cr_dict)


def build_helpers(df,d, cust_master, agent_master):
    
    import time
    start = time.time()

    # TODO to be changed
    d['capa_min'],d['capa_max'] = minMaxCapaAgt(d['AP'],d['A'])
    print('capa_min',np.round(time.time() - start,0))

    # ok for MVP
    d['expectedValue'] = compute_expected_values(df,d['CP'])
    print('expectedValue',np.round(time.time() - start,0))
    
    # TODO to be changed  ==> P1
    #d['affinity'] = compute_affinity(d['CA'])
    #print('affinity',np.round(time.time() - start,0))
    
    #affinity est
    d['affinity'] = affinity_est(df, cust_master, agent_master)
    
    # TODO to be changed
    d['priority'] =  compute_agentPriority(df,d['CA'])
    print('priority',np.round(time.time() - start,0))
    
    # done
    d['APdomain'],d['CAdomain'],d['Cdomain'] = build_duo_domain(d['CAP'])
    print('APdomain',np.round(time.time() - start,0))
    
    # TODO to be changed
    d['min_max_product'] = product_min_max(d['P'])
    print('min_max_product',np.round(time.time() - start,0))
    
    # done
    d['CPattached'],d['CPorphan'] = build_CPattached_CPoprhan(df,d['CAP'])
    print('CPattached',np.round(time.time() - start,0))
    
    # TODO to be changed
    d['Anowarming'],d['Awarming_compuls'] = build_Awarming(d['A'])
    print('Anowarming',np.round(time.time() - start,0))
    
    # TODO to be changed ==> P1
    #d['warmingScore'] = build_warmingScore(d['CwAP'])
    #print('warmingScore',np.round(time.time() - start,0))
                         
    d['warmingScore'] = warming_score_calc(d['CwAP'], df, cr_df)
    print('warmingScore',np.round(time.time() - start,0))

    # TODO to be changed
    d['warming_capa'] = max_warming_type()
    print('warming_capa',np.round(time.time() - start,0))
    
    # done
    d['CwAPdomain'] =  build_warming_domains(d['CwAP'],d['AT'])
    print('CwAPdomain',np.round(time.time() - start,0))

    #done
    d['CwPdomain'] = build_CwPdomain(d['CwAP'])
    print('CwPdomain',np.round(time.time() - start,0))
    
    
    return d

#from __future__ import print_function
from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np

'''
----------------------Create the mip solver with the CBC backend----------------------
'''

def launcher(name='solver', solver_type = 'CBC'):
    '''
    name: string assignment for indicating solver type
    '''
    solver = pywraplp.Solver.CreateSolver(name, solver_type)
    #logger.info('instantiated solver')
    
    return solver


'''
-------------------
1) Instantate variables for CAP[c,a,p] and capa[a]
-------------------
'''
def instantiate_vars(CAP,A, solver):
    #instantiating x[c,a,p]
    x={}
    for (c,a,p) in CAP:
        x[c,a,p] = solver.IntVar(0,1,'')

    #instantiating capa[a]
    capa={}
    for a in A:
        capa[a] = solver.IntVar(0,np.inf,'')
        
    return x, capa

'''
-------------------
2) Set constraints
-------------------
'''
def constraint_singular(C,APdomain, x, solver):
    '''
    Cconstraint 1: 1 combintion of customer --> agent x product
    '''    
    for c in C:
        solver.Add(solver.Sum([x[c, a, p] for a,p in APdomain[c]]) <= 1)
        

def constraint_min_max(P,CAdomain,min_max_product, x, solver):
    '''
    Constraint 2: min / max for each product
    '''    
    for p in P:
        solver.Add(solver.Sum([x[c, a, p] for c,a in CAdomain[p]]) >= min_max_product[p]['min']  )
        solver.Add(solver.Sum([x[c, a, p] for c,a in CAdomain[p]]) <= min_max_product[p]['max']  )


def constraint_agent_min_cap(AP,Cdomain,capa_min,x,solver):
    '''
    Constraint 3: capacity agent higher than minimum / maximum
    '''
    for a,p in AP:
        solver.Add(solver.Sum([x[c, a, p] for c in Cdomain[a,p]]) >= capa_min[a,p] )


def constraint_rem_capa(A,capa_max,CPattached,x,capa,solver):
    '''
    Constraint 4: definition of capa
    '''
    for a in CPattached:
        solver.Add(  capa_max[a] - solver.Sum([x[c, a, p] for c,p in CPattached[a]])   <= capa[a] )


def constraint_max_orphan(A,CPorphan,x,capa,solver):
    '''
    Constraint6: max orphan leads per agent per use case
    ''' 
    for a in CPorphan:
        solver.Add(solver.Sum([x[c, a, p] for c,p in CPorphan[a] ]) <= capa[a])




'''
-------------------
3) Launch Solver
-------------------
'''

def solver_launch(CAP,A,expectedValue,affinity,priority, x, capa, solver,solver_time_min = 1):

    '''
    -------------------
    Solver status
    -------------------
    dict_status = {
                    pywraplp.Solver.OPTIMAL: 'Optimal',
                    pywraplp.Solver.FEASIBLE: 'Feasible',
                    pywraplp.Solver.INFEASIBLE: 'Infeasible',
                    pywraplp.Solver.UNBOUNDED: 'Unbounded',
                    pywraplp.Solver.ABNORMAL: 'Abnormal',
                    pywraplp.Solver.NOT_SOLVED: 'Not solved'
                  }
    '''
    """
    Add objective function
    """
    # TODO - set otpimization limits / GAPS ?
    #solver.SetTimeLimit(1000*solver_time_min)
    solver.SetTimeLimit(60*1000*solver_time_min)
    #print('setting time limite to',60*solver_time_min,'s')

    solver.Maximize(solver.Sum([x[c,a,p]*expectedValue[c,p] for c,a,p in CAP ]) \
                  + solver.Sum([x[c,a,p] * affinity[c,a] for c,a,p in CAP])\
                  + solver.Sum([-capa[a] for a in A]) )


    status = solver.Solve()
    print('status = {0}'.format(status))

    '''
    -------------------
    4) Print output callback
    -------------------
    '''
    out_d = []
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print('Total revenue = ', solver.Objective().Value(), '\n')
        out_d  = [ [c,a,p,expectedValue[c,p],affinity[c,a],priority[c,a],] for (c,a,p) in CAP if x[c,a,p].solution_value() ==1 ]
        columns = ['customerseqid','agentseqid','usecase','expectedValue','affinity','priority']
        out_d = pd.DataFrame(out_d,columns=columns)

    return out_d


'''
--------------------------------------
post processor for optimizer 2
--------------------------------------
'''


def calculate_allocated(df,A,Cw,CwAP):
    '''
    retuns allocation matrix with # of leads allocated to each agent seqid
    '''
    # total allocated
    allocated = df.groupby('agentseqid')[['customerseqid']].nunique()['customerseqid'] .to_dict()
    for a in A:
        if a not in allocated:
            allocated[a] = 0

    # total allocated in warming
    allocated_w = df [df.customerseqid.isin(Cw)].groupby('agentseqid')[['customerseqid']].nunique()['customerseqid'] .to_dict()
    for a in A:
        if a not in allocated_w:
            allocated_w[a] = 0    

    # selecting all records in CwAP but with no allocation ==> TO put them to 0 warming
    cols = ['customerseqid','agentseqid','usecase']
    no_warming = pd.DataFrame([[x[0],x[1],x[2]] for x in CwAP],columns = cols)

    # anti join
    no_warming = no_warming.merge(df[cols], how = 'outer', indicator = True)
    no_warming = no_warming[(no_warming._merge == 'left_only')].drop('_merge', axis = 1)

    # records now
    no_warming = list(no_warming.to_records(index=False))
   
    return allocated,no_warming,allocated_w


'''
variables
'''

##instantiation
def instantiate_warming(CwAP, solver):
    '''
    #instantiating y[cw,a,p_allocated]
    '''
    y={}
    for [c,a,p] in CwAP:
        y[c,a,p] = solver.IntVar(0,1,'')
        
    return y

'''
constraints
'''

#constraint #1 - singularity of y
def constraint_warming_singular(Cw,APdomain, y, solver):
    '''
    Cconstraint 1: one warming at a time
    '''    
    for c in Cw:
        solver.Add(solver.Sum([y[c, a, p] for a,p in APdomain[c]]) <= 1)

#constraint #2
def constraint_cap_rfs_agency(CwAPdomain,warming_capa,y, solver):
    '''
    Constraint 2 : max capacity by rfs/agency
    '''
    for t in ['rfs', 'agency'] :
        solver.Add(solver.Sum([y[c, a, p] for [c,a,p] in CwAPdomain[t]]) <= warming_capa[t] )
      

#constraint 3: ensuring agents who have opted out of warming do not get warmed leads
def constraint_no_warming_agents(Anowarming,Awarming_compuls,CwPdomain,allocated_w,y,solver):
    '''
    #constraining no warming leads to be assigned for agents who have opted out
    # consraining compulsory warming - y = 1 (all warmed)
    '''
    # forbidden
    for a in Anowarming:
        if a not in CwPdomain: continue
        solver.Add(solver.Sum([y[c, a, p]  for c,p in CwPdomain[a]]) <= 0)

    # compulsory
    for a in Awarming_compuls:
        if a not in CwPdomain: continue 
        solver.Add(solver.Sum([y[c, a, p]  for c,p in CwPdomain[a] ]) >= allocated_w[a])  

# constraint 4
def constraint_no_warming_no_allocation(no_warming,y,solver):
    """
    # TODO - Force all y = 0 if x = 0 ==> PAUL
    Forcing y = 0 if x = 0
    """
    solver.Add(solver.Sum([y[c, a, p]  for c,a,p in no_warming ]) <= 0)


#solve obj for stage 2 warming optimiser
def solve_warming(out_d, CwAP, expectedValue,warmingScore, allocated, y, solver,solver_time_min=2):
    '''
    -------------------
    4) Launch Solver for stage 2 optimiser
    -------------------
    '''
    # TODO - limits
    solver.SetTimeLimit(60*1000*solver_time_min)
    # print('setting time limite to',60*solver_time_min,'s')

    solver.Maximize(solver.Sum( [y[c,a,p]*expectedValue[c,p]*warmingScore[c,a,p] for c,a,p in CwAP]) \
                 + solver.Sum( [y[c,a,p] * allocated[a]   for c,a,p in CwAP]) )



    '''
    -------------------
    Solver status
    -------------------
    dict_status = {
                    pywraplp.Solver.OPTIMAL: 'Optimal',
                    pywraplp.Solver.FEASIBLE: 'Feasible',
                    pywraplp.Solver.INFEASIBLE: 'Infeasible',
                    pywraplp.Solver.UNBOUNDED: 'Unbounded',
                    pywraplp.Solver.ABNORMAL: 'Abnormal',
                    pywraplp.Solver.NOT_SOLVED: 'Not solved'
                  }
    '''

    status = solver.Solve()
    print('status = {0}'.format(status))

    '''
    -------------------
    5) Print output callback
    -------------------
    '''

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        
        print('Total revenue = ', solver.Objective().Value(), '\n')
        out_w  = [ [c,a,p,1,warmingScore[(c,a,p)]] for (c,a,p) in CwAP if y[c,a,p].solution_value() ==1 ]

        columns = ['customerseqid','agentseqid','usecase','warming','warmingScore']
        out_w = pd.DataFrame(out_w,columns=columns)

        # merge output
        out_d = out_d.merge(out_w,on=['customerseqid','agentseqid','usecase'],how='left')
        out_d['warming'].fillna(0,inplace=True)   
                    
    return out_d





#############################################################
# main
#############################################################

#1. allocation
def run_opt_allocation(CAP,
                       AP,
                       C,
                       A,
                       P,
                       APdomain,
                       CAdomain,
                       Cdomain,
                       CPattached,
                       CPorphan,
                       min_max_product,
                       capa_min,
                       capa_max,
                       expectedValue,
                       affinity,
                       priority,
                       solver_time_min
                      ):
    import time
    start = time.time()

    solver = launcher()
    print('init 1',np.round(time.time() - start,0))

    x,capa = instantiate_vars(CAP,A, solver)
    print('init 2',np.round(time.time() - start,0))

    constraint_singular(C,APdomain, x, solver)
    print('const 1',np.round(time.time() - start,0))

    constraint_min_max(P,CAdomain,min_max_product, x, solver)
    print('const 2',np.round(time.time() - start,0))

    constraint_agent_min_cap(AP,Cdomain,capa_min,x,solver)
    print('const 3',np.round(time.time() - start,0))

    constraint_rem_capa(A,capa_max,CPattached,x,capa,solver)
    print('const 4',np.round(time.time() - start,0))

    constraint_max_orphan(A,CPorphan,x,capa,solver)
    print('const 5',np.round(time.time() - start,0))

    out_d = solver_launch(CAP,A,expectedValue,affinity,priority, x, capa, solver,solver_time_min)
    print('solv 1',np.round(time.time() - start,0))
    
    return out_d

# 2. warming
def run_opt_warming   (out_d,
                       CwAP,
                       Cw,
                       A,
                       APdomain,
                       Anowarming,
                       Awarming_compuls,
                       CwAPdomain,
                       CwPdomain,
                       warming_capa,
                       expectedValue,
                       warmingScore,
                       solver_time_min
                      ):
    import time
    start = time.time()

    solver = launcher()
    print('init 1',np.round(time.time() - start,0))
    
    allocated,no_warming,allocated_w = calculate_allocated(out_d,A,Cw,CwAP)
    print('init 2',np.round(time.time() - start,0))
    
    y = instantiate_warming(CwAP, solver)
    print('init 3',np.round(time.time() - start,0))

    constraint_warming_singular(Cw,APdomain, y, solver)
    print('const 1',np.round(time.time() - start,0))

    constraint_cap_rfs_agency(CwAPdomain,warming_capa,y, solver)
    print('const 2',np.round(time.time() - start,0))

    constraint_no_warming_agents(Anowarming,Awarming_compuls,CwPdomain,allocated_w,y,solver)
    print('const 3',np.round(time.time() - start,0))

    constraint_no_warming_no_allocation(no_warming,y,solver)
    print('const 4',np.round(time.time() - start,0))
    
    out_d = solve_warming(out_d, CwAP, expectedValue,warmingScore, allocated, y, solver,solver_time_min)
    print('solv 1',np.round(time.time() - start,0))
    
    return out_d



#native libraries
import pandas as pd
import numpy as np
import os
import sys
import warnings 
import logging
import configparser
warnings.filterwarnings("ignore")

#sub utils
import datetime
from logger import logger

#opti functions (google or tools)

#from __future__ import print_function
from ortools.linear_solver import pywraplp


# In[2]:
'''

#custom modules
import os, imp
#os.chdir('../../src/opti')
import build_domains as bd
import build_helper as bh
import build_opt as bo
import model_preprocessing as mp


# # Reading path


export_name = "s3://data-analytics-emr-data-sit-income-com-sg/opti/in/paul_test_df"
df = mp.s3_parquet_to_pandas(export_name)

test = True
if test:
    df = df.sample(n=200)
print('!!!!!!!!!!!!!!')
print('working with',len(df),'rows')


# # Build helpers and domains

imp.reload(bd)
imp.reload(bh)


import time
start = time.time()

d = build_domains(df)
print('domains',np.round(time.time() - start,0))

d = build_helpers(df,d)
print('helpers',np.round(time.time() - start,0))



# # Running opt - Allocation


imp.reload(bo)

out_d = bo.run_opt_allocation( d['CAP'],
                               d['AP'],
                               d['C'],
                               d['A'],
                               d['P'],
                               d['APdomain'],
                               d['CAdomain'],
                               d['Cdomain'],
                               d['CPattached'],
                               d['CPorphan'],
                               d['min_max_product'],
                               d['capa_min'],
                               d['capa_max'],
                               d['expectedValue'],
                               d['affinity'],
                               d['priority']
                          )


# # Run opt - warming
imp.reload(bo)
out_f = bo.run_opt_warming (out_d,
                           d['CwAP'],
                           d['Cw'],
                           d['A'],
                           d['APdomain'],
                           d['Anowarming'],
                           d['Awarming_compuls'],
                           d['CwAPdomain'],
                           d['CwPdomain'],
                           d['warming_capa'],
                           d['expectedValue'],
                           d['warmingScore']
                          )

from models.config import ModelPipelineConfig

#native libraries
import pandas as pd
import numpy as np
import os
from os import path
import sys
import warnings
import logging
import configparser
warnings.filterwarnings("ignore")

#sub utils
import datetime
#from util.io_layer import io_ds as io

#file formats
import yaml
import re

import models.model_preprocessing as mp
import models.model_data_generator as mdg
import models.model_training as mt
import models.model_scoring as ms


# display formatter
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)

# setup custom logger
logger = logging.getLogger(__name__)

# TODO
base_dir = os.getcwd()
config_dir = base_dir + "/configs"

# model_configs = [f for f in os.listdir(config_dir) if path.isfile(path.join(config_dir, f))]
'''

def train_score(mpc):
    model_name = mpc.model_name
    today_dt = datetime.datetime.today().strftime('%Y-%m-%d')#config instantiation
    config = mpc.external_config

    # identifiers
    primary_target = config['identifiers']['primary_target']
    time_field = config['identifiers']['time_fields']
    id_cols = config['identifiers']['id_columns'].split(',')
    primary_customer = config['identifiers']['primary_cust']

    # params
    seed = config['run_configs']['seed_val']
    date_impute_logic = config['run_configs'].get('date_impute_logic', "max")
    holdout_months = config['run_configs']['holdout_months']
    estimators = config['run_configs']['estimators']
    estimators = ['XGB'] #TODO remove
    calibration_method = config['run_configs'].get("calibration_method", None)

    base_df = mp.processing_pipeline(run_tag=model_name,
                                     run_date=today_dt,
                                     id_columns=id_cols,
                                     holdout_months=holdout_months,
                                     time_field='monthyearid',
                                     primary_target='target_flag',
                                     primary_customer='customerseqid',
                                     continuous_date_identifier='days_',
                                     date_impute_strategy=date_impute_logic)
    # TODO separate out funciton to remove scoring df as it will be its own pipeline

    X_train, X_test, y_train, y_test, holdout_df, for_scoring_df = \
        mdg.model_data_generator_sampling(base_df,
                                          target_label='target_flag',
                                          primary_customer='customerseqid',
                                          time_key='monthyearid',
                                          id_cols=id_cols,
                                          today_dt=today_dt,
                                          n_feats_REF=300,
                                         seed=42,
                                          holdout_column="holdout",
                                          scoring_data_column="for_scoring",
                                          downsampling=1,
                                          exclude_cols=[],
                                          rfe_test_size=0.75,
                                          model_test_size=0.3,
                                          corr_threshold=0.8,
                                          feature_export_path='output/')

    if calibration_method:
        io.put(f"holdout_df_{model_name}", holdout_df)
        io.put(f"scoring_df_{model_name}", for_scoring_df)


    for est in estimators:
        mt.bayes_model_core(X_train, X_test, y_train, y_test, today_dt, ITERATIONS = 10,
                            run_tag = model_name, est = est, cv_splits = 3, calibration_method=calibration_method)

    # TODO voting when more than 1 estimator
    voting_model_out = mt.voting_ensemble(X_train, y_train, X_test, y_test,
                                          model_path='output/',
                                          today_dt=today_dt, est='votingmodel',
                                          run_tag=model_name,
                                          export_path='output/',
                                          estimators=estimators
                                          )
    #measure
    ho_comparison, best_model_ho = mt.holdout_stats(holdoutdf=holdout_df, idcols=id_cols,
                                                    primary_target='target_flag',
                                                    import_path='output/',
                                                    today_dt=today_dt,
                                                    run_tag=model_name,
                                                    reference_percentile=3,
                                                    estimators=estimators)

    # TODO for_scoring_df Separate train with date parameters
    scored_leads = ms.score_leads(for_scoring_df, best_model=best_model_ho, idcols=id_cols, \
                                  primary_key='customerseqid', \
                                  primary_target='target_flag', \
                                  export_path_local='output/', run_tag=model_name, today_dt=today_dt)



if __name__ == "__main__":
    # train_score(ModelPipelineConfig("configs/xsell_pa_config.yml"))
    # train_score(ModelPipelineConfig("configs/upsell_life_single_config.yml"))
    # train_score(ModelPipelineConfig("configs/upsell_life_multi_config.yml"))
    # train_score(ModelPipelineConfig("configs/xsell_life_config.yml"))
    # train_score(ModelPipelineConfig("configs/xsell_health_config.yml"))
    # train_score(ModelPipelineConfig("configs/fpp_config.yml"))
    # train_score(ModelPipelineConfig("configs/categ_ilp_config.yml"))
    # train_score(ModelPipelineConfig("configs/categ_saving_config.yml"))
    # train_score(ModelPipelineConfig("configs/categ_term_config.yml"))
    # train_score(ModelPipelineConfig("configs/categ_wholelife_config.yml"))
    train_score(ModelPipelineConfig("configs/xsell_home_config.yml")) # TODO

