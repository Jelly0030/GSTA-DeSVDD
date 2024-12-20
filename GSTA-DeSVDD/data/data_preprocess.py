import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler


#--------------swat--------------#
def swat_dataset(n):
    df1 = pd.read_csv("SWaT_Normal.csv", low_memory=False, header=1)
    df2 = pd.read_csv("SWaT_Attack.csv", low_memory=False, header=1)
    x_train = df1.iloc[:, 1:-1]
    x_train_names = x_train.columns.tolist()
    y_train = df1.iloc[:, -1].replace({'Normal': 1, 'Attack': 0, 'A ttack': 0}).astype(int)
    x_test = df2.iloc[:, 1:-1]
    x_test_names = x_test.columns.tolist()
    y_test = df2.iloc[:, -1].replace({'Normal': 1, 'Attack': 0, 'A ttack': 0}).astype(int)
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    scaler.fit(x_test)
    x_train = pd.DataFrame(MinMaxScaler().fit_transform(x_train), columns=x_train_names)
    x_test = pd.DataFrame(MinMaxScaler().fit_transform(x_test), columns=x_test_names)
    data_train = pd.concat([x_train, pd.DataFrame(y_train)], axis=1)
    data_test = pd.concat([x_test, pd.DataFrame(y_test)], axis=1)
    np.random.seed(23)
    data_train_index = random.randint(0, len(data_train) - n)
    data_train = data_train[data_train_index: data_train_index + n]
    data_test_index = random.randint(0, len(data_test) - n)
    data_test = data_test[data_test_index: data_test_index + n]
    x_train = data_train.iloc[:, :-1].values.astype(np.float32)
    y_train = data_train.iloc[:, -1].values.astype(np.int32)
    x_test = data_test.iloc[:, :-1].values.astype(np.float32)
    y_test = data_test.iloc[:, -1].values.astype(np.int32)
    return x_train, y_train, x_test, y_test

#--------------wadi--------------#
def wadi(n):
    df1 = pd.read_csv("WADI_14days_new.csv.csv",low_memory=False, header=1)
    df2 = pd.read_csv("WADI_attackdataLABLE.csv",low_memory=False, header=1)
    df1.fillna(0, inplace=True)
    df2.fillna(0, inplace=True)
    x_train = df1.iloc[:, 3:-1]
    x_train_names = x_train.columns.tolist()
    x_test = df2.iloc[:, 3:-1]
    x_test_names = x_test.columns.tolist()
    y_train = df1.iloc[:, -1]
    y_test = df2.iloc[:, -1].replace(-1, 0) #1:No Attack, -1:Attack
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    scaler.fit(x_test)
    x_train = pd.DataFrame(MinMaxScaler().fit_transform(x_train), columns=x_train_names)
    x_test = pd.DataFrame(MinMaxScaler().fit_transform(x_test), columns=x_test_names)
    data_train = pd.concat([x_train, pd.DataFrame(y_train)], axis=1)
    data_test = pd.concat([x_test, pd.DataFrame(y_test)], axis=1)
    np.random.seed(23)
    data_train_index = random.randint(0, len(data_train) - n)
    data_train = data_train[data_train_index: data_train_index + n]
    data_test_index = random.randint(0, len(data_test) - n)
    data_test = data_test[data_test_index: data_test_index + n]
    np.random.seed(23)
    data_train_index = random.randint(0, len(data_train) - n)
    data_train = data_train[data_train_index: data_train_index + n]
    data_test_index = random.randint(0, len(data_test) - n)
    data_test = data_test[data_test_index: data_test_index + n]
    x_train = data_train.iloc[:, :-1].values.astype(np.float32)
    y_train = data_train.iloc[:, -1].values.astype(np.int32)
    x_test = data_test.iloc[:, :-1].values.astype(np.float32)
    y_test = data_test.iloc[:, -1].values.astype(np.int32)
    return x_train, y_train, x_test, y_test


#--------------kddcup99--------------#
def get_train(*args):
    return _get_adapted_dataset("train")
def get_test(*args):
    return _get_adapted_dataset("test")
def get_shape_input():
    return (None, 121)
def get_shape_label():
    return (None,)
def _get_adapted_dataset(split):
    dataset = kdd_dataset()
    key_img = 'x_' + split
    key_lbl = 'y_' + split
    if split != 'train':
        dataset[key_img], dataset[key_lbl] = _adapt(dataset[key_img],
                                                    dataset[key_lbl])

    return (dataset[key_img], dataset[key_lbl])
def _encode_text_dummy(df, name):
    dummies = pd.get_dummies(df.loc[:, name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df.loc[:, dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)
def _to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    dummies = df[target]
    return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
def _col_names():
    return ["duration", "protocol_type", "service", "flag", "src_bytes",
            "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
            "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
            "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
            "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
def _adapt(x, y, rho=0.2):
    rng = np.random.RandomState(42)  # seed shuffling
    inliersx = x[y == 1]
    inliersy = y[y == 1]
    outliersx = x[y == 0]
    outliersy = y[y == 0]
    size_outliers = outliersx.shape[0]
    inds = rng.permutation(size_outliers)
    outliersx, outliersy = outliersx[inds], outliersy[inds]
    size_test = inliersx.shape[0]
    out_size_test = int(size_test * rho / (1 - rho))
    outestx = outliersx[:out_size_test]
    outesty = outliersy[:out_size_test]
    testx = np.concatenate((inliersx, outestx), axis=0)
    testy = np.concatenate((inliersy, outesty), axis=0)
    size_test = testx.shape[0]
    inds = rng.permutation(size_test)
    testx, testy = testx[inds], testy[inds]
    return testx, testy
def kdd_dataset(n):
    col_names = _col_names()
    df = pd.read_csv("kddcup.data_10_percent_corrected",
                     header=None, names=col_names)[:n]
    text_l = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']

    for name in text_l:
        _encode_text_dummy(df, name)

    labels = df['label'].copy()
    labels[labels != 'normal.'] = 0
    labels[labels == 'normal.'] = 1

    df['label'] = labels

    df_train = df.sample(frac=0.5, random_state=42)
    df_test = df.loc[~df.index.isin(df_train.index)]

    x_train, y_train = _to_xy(df_train, target='label')
    y_train = y_train.flatten().astype(int)
    x_test, y_test = _to_xy(df_test, target='label')
    y_test = y_test.flatten().astype(int)

    x_train = x_train[y_train == 1]
    y_train = y_train[y_train == 1]

    x_test = MinMaxScaler().fit_transform(x_test)
    x_train = MinMaxScaler().fit_transform(x_train)

    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    return x_train, y_train, x_test, y_test

