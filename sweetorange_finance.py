import re
import numpy as np
import pandas as pd 
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import plot_importance
import xgboost as xgb
import warnings
import matplotlib.pyplot as plt
# import time
# start =time.clock()
#
# end = time.clock()
# print('Running time: %s Seconds'%(end-start))
# np.set_printoptions(threshold='nan')
# # pd.set_option('display.width', None)
# # pd.set_option('display.max_rows', None)
warnings.filterwarnings('ignore')

n_splites = 5   # 交叉验证行数
seed = 42
params = {
    'objective': 'multi:softmax',
    'num_class': 2,  # –让XGBoost采用softmax目标函数处理多分类问题，同时需要设置参数num_class（类别个数）
    'gamma': 0.005,  # 在树的叶节点上进行进一步分区所需的最小损失减少量。 算法越大，越保守。
#     'eval_metric':'merror',
    'max_depth': 11,  # 数的最大深度。缺省值为6。
    'lambda': 0.8,  # L2 正则的惩罚系数
    'subsample': 0.9, #用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中随机的抽取出50%的子样本建立树模型，这能够防止过拟合。
# 'gpu_id':0,
# 'updater':'grow_gpu',
# 'tree_method':'gpu_hist',
    'colsample_bytree': 0.8,  # 在建立树时对特征采样的比例。缺省值为1
    'scale_pos_weight': 1.5,  # 是用来调节正负样本不均衡问题的，用助于样本不平衡时训练的收敛。
    'min_child_weight': 0.7,  # 孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。
    'silient': 1,  #
    'eta': 0.1,  # 学习率
    'seed': 1000,  # 随机数种子
    'n_estimators': 150,  # 随机森林树数量
    'nthread': 6
    }

# train_data = pd.read_csv('train_all.csv', delimiter=',', low_memory=False)
train_data = pd.read_csv('tag_train_new.csv', delimiter=',', low_memory=False)
train_data_opera = pd.read_csv('operation_train_new.csv', delimiter=',', low_memory=False)
train_data_trans = pd.read_csv('transaction_train_new.csv', delimiter=',', low_memory=False)

train_trans_uid_counts = dict(train_data_trans['UID'].value_counts())
train_data_trans['time'] = train_data_trans['time'].apply(lambda x: float(x.replace(':', ''))/240000)
train_data['trans_sum'] = 0
train_data['trans_all_money'] = 0
train_data['trans_channel'] = 0
# train_data['trans_merchant'] = np.nan
train_data['trans_amt_src1'] = np.nan

# train_data['trans_market_code'] = np.nan
# train_data['trans_acc_id1'] = np.nan
# train_data['trans_ip1_sub'] = np.nan



#  ————————————————————训练操作 交易 分割线————————————————————————————————
train_opera_uid_counts = dict(train_data_opera['UID'].value_counts())
opera_succe_counts = dict(train_data_opera.loc[train_data_opera.success == 1, 'UID'].value_counts())

train_data_opera['time'] = train_data_opera['time'].apply(lambda x: float(x.replace(':', ''))/240000)
train_data_opera['ip'] = train_data_opera['ip1'].fillna('@').map(str) + train_data_opera['ip2'].fillna('@').map(str)
train_data_opera['ip'] = train_data_opera['ip'].apply(lambda x: x.replace('@', ''))

train_data['opera_sum'] = 0  # 操作总次数
train_data['opera_sum_not_succe'] = 0  # 操作不成功次数
# train_data['opera_average_time'] = np.nan  # 操作平均时间段  注意有空值后面再处理！！！！！！！！！！！
# train_data['opera_os'] = np.nan
train_data['trans_recency'] = np.nan# 注意有空值后面再处理！！！！！！！！！！！
# train_data['opera_device1'] = np.nan  # 注意有空值后面再处理！！！！！！！！！！！
# train_data['opera_device2'] = np.nan  # 注意有空值后面再处理！！！！！！！！！！！
# train_data['opera_wifi'] = np.nan  # 注意有空值后面再处理！！！！！！！！！！！
# train_data['opera_geo_code'] = np.nan  # 注意有空值后面再处理！！！！！！！！！！！
# train_data['opera_ip'] = np.nan  # 注意有空值后面再处理！！！！！！！！！！！

train_data_opera = train_data_opera.reset_index()
opera_user_id = list(train_data_opera['UID'])
trans_user_id = list(train_data_trans['UID'])


userid = train_data['UID']
for uid in userid:
    if uid in opera_user_id:
        opera_sum = train_opera_uid_counts[uid]
        succe_sum = uid in opera_succe_counts and opera_succe_counts[uid] or 0
        not_succe = (train_opera_uid_counts[uid] - succe_sum)/10
        # opera_average_time = train_data_opera.loc[train_data_opera.UID == uid, 'time'].mean()
        os = (train_data_opera.loc[train_data_opera.UID == uid, 'os'].value_counts()).idxmax()
        # device1 = (any(train_data_opera.loc[train_data_opera.UID == uid, 'device1'].value_counts()) and (train_data_opera.loc[train_data_opera.UID == uid, 'device1'].value_counts()).idxmax()) or np.nan
        # device2 = (any(train_data_opera.loc[train_data_opera.UID == uid, 'device2'].value_counts()) and (train_data_opera.loc[train_data_opera.UID == uid, 'device2'].value_counts()).idxmax()) or np.nan
        # wifi = (any(train_data_opera.loc[train_data_opera.UID == uid, 'wifi'].value_counts()) and (train_data_opera.loc[train_data_opera.UID == uid, 'wifi'].value_counts()).idxmax()) or np.nan
        # ip = (any(train_data_opera.loc[train_data_opera.UID == uid, 'ip'].value_counts()) and (train_data_opera.loc[train_data_opera.UID == uid, 'ip'].value_counts()).idxmax()) or np.nan


    if uid in trans_user_id:
        trans_sum = train_trans_uid_counts[uid]
        # trans_average_time = train_data_trans.loc[train_data_trans.UID == uid, 'time'].mean()
        trans_all_money = train_data_trans.loc[train_data_trans.UID == uid, 'trans_amt'].sum()
        trans_channel = any(train_data_trans.loc[train_data_trans.UID == uid, 'channel'].value_counts()) and train_data_trans.loc[train_data_trans.UID == uid, 'channel'].value_counts().idxmax() or np.nan
        # trans_merchant = any(train_data_trans.loc[train_data_trans.UID == uid, 'merchant'].value_counts()) and train_data_trans.loc[train_data_trans.UID == uid, 'merchant'].value_counts().idxmax() or np.nan
        trans_amt_src1 = any(train_data_trans.loc[train_data_trans.UID == uid, 'amt_src1'].value_counts()) and train_data_trans.loc[train_data_trans.UID == uid, 'amt_src1'].value_counts().idxmax() or np.nan
        # trans_market_code = any(train_data_trans.loc[train_data_trans.UID == uid, 'market_code'].value_counts()) and train_data_trans.loc[train_data_trans.UID == uid, 'market_code'].value_counts().idxmax() or np.nan
        trans_recency = any(train_data_trans.loc[train_data_trans.UID == uid, 'day']) and train_data_trans.loc[train_data_trans.UID == uid, 'day'].max() or 0
        # trans_acc_id1 = any(train_data_trans.loc[train_data_trans.UID == uid, 'acc_id1'].value_counts()) and train_data_trans.loc[train_data_trans.UID == uid, 'acc_id1'].value_counts().idxmax() or np.nan
        # trans_ip1_sub = any(train_data_trans.loc[train_data_trans.UID == uid, 'ip1_sub'].value_counts()) and train_data_trans.loc[train_data_trans.UID == uid, 'ip1_sub'].value_counts().idxmax() or np.nan
        trans_average_money = train_data_trans.loc[train_data_trans.UID == uid, 'trans_amt'].mean()

    train_data.loc[(train_data.UID == uid), ['opera_sum', 'opera_sum_not_succe', 'trans_sum', 'trans_all_money', 'trans_channel', 'trans_amt_src1', 'trans_recency']] = \
        [opera_sum, not_succe, trans_sum, trans_all_money, trans_channel,  trans_amt_src1, trans_recency]  # 操作的总次数作为一列

    # ——————————————开始属于测试--分割线————————————————————————————————————————————————————
test_data = pd.read_csv('提交样例.csv', delimiter=',', low_memory=False)
test_data_opera = pd.read_csv('operation_round1_new.csv', delimiter=',', low_memory=False)
test_data_trans = pd.read_csv('transaction_round1_new.csv', delimiter=',', low_memory=False)
test_trans_uid_counts = dict(test_data_trans['UID'].value_counts())
test_data_trans['time'] = test_data_trans['time'].apply(lambda x: float(x.replace(':', '')) / 240000)
test_data['trans_sum'] = 0
test_data['trans_all_money'] = 0
test_data['trans_channel'] = 0
#test_data['trans_merchant'] = np.nan
test_data['trans_amt_src1'] = np.nan
#test_data['trans_market_code'] = np.nan
# test_data['trans_acc_id1'] = np.nan
# test_data['trans_ip1_sub'] = np.nan

#  ———————————————————— 操作交易 分割线————————————————————————————————
test_opera_uid_counts = dict(test_data_opera['UID'].value_counts())
test_opera_succe_counts = dict(test_data_opera.loc[test_data_opera.success == 1, 'UID'].value_counts())


test_data_opera['time'] = test_data_opera['time'].apply(lambda x: float(x.replace(':', ''))/240000)
test_data_opera['ip'] = test_data_opera['ip1'].fillna('@').map(str) + test_data_opera['ip2'].fillna('@').map(str)
test_data_opera['ip'] = test_data_opera['ip'].apply(lambda x: x.replace('@', ''))

test_data['opera_sum'] = 0  # 操作总次数
test_data['opera_sum_not_succe'] = 0  # 操作不成功次数
# test_data['opera_average_time'] = np.nan  # 操作平均时间段  注意有空值后面再处理！！！！！！！！！！！
# test_data['opera_os'] = np.nan  # 注意有空值后面再处理！！！！！！！！！！！
test_data['trans_recency'] = np.nan
# test_data['opera_device1'] = np.nan  # 注意有空值后面再处理！！！！！！！！！！！
# test_data['opera_device2'] = np.nan  # 注意有空值后面再处理！！！！！！！！！！！
# test_data['opera_wifi'] = np.nan  # 注意有空值后面再处理！！！！！！！！！！！
#test_data['opera_geo_code'] = np.nan  # 注意有空值后面再处理！！！！！！！！！！！
# test_data['opera_ip'] = np.nan  # 注意有空值后面再处理！！！！！！！！！！！

test_data_opera = test_data_opera.reset_index()
opera_user_id = list(test_data_opera['UID'])
trans_user_id = list(test_data_trans['UID'])

userid = test_data['UID']
for uid in userid:
    if uid in opera_user_id:
        opera_sum = test_opera_uid_counts[uid]
        succe_sum = uid in opera_succe_counts and opera_succe_counts[uid] or 0
        not_succe = (test_opera_uid_counts[uid] - succe_sum) / 10
        # opera_average_time = test_data_opera.loc[test_data_opera.UID == uid, 'time'].mean()
        os = (test_data_opera.loc[test_data_opera.UID == uid, 'os'].value_counts()).idxmax()
        # device1 = (any(test_data_opera.loc[test_data_opera.UID == uid, 'device1'].value_counts()) and (test_data_opera.loc[test_data_opera.UID == uid, 'device1'].value_counts()).idxmax()) or np.nan
        # device2 = (any(test_data_opera.loc[test_data_opera.UID == uid, 'device2'].value_counts()) and (test_data_opera.loc[test_data_opera.UID == uid, 'device2'].value_counts()).idxmax()) or np.nan
        # wifi = (any(test_data_opera.loc[test_data_opera.UID == uid, 'wifi'].value_counts()) and (test_data_opera.loc[test_data_opera.UID == uid, 'wifi'].value_counts()).idxmax()) or np.nan
        # ip = (any(test_data_opera.loc[test_data_opera.UID == uid, 'ip'].value_counts()) and (test_data_opera.loc[test_data_opera.UID == uid, 'ip'].value_counts()).idxmax()) or np.nan

    if uid in trans_user_id:
        trans_sum = test_trans_uid_counts[uid]
         # trans_average_time = test_data_trans.loc[test_data_trans.UID == uid, 'time'].mean()
        trans_all_money = (test_data_trans.loc[test_data_trans.UID == uid, 'trans_amt'].sum())/10000
        trans_channel = any(test_data_trans.loc[test_data_trans.UID == uid, 'channel'].value_counts()) and test_data_trans.loc[test_data_trans.UID == uid, 'channel'].value_counts().idxmax() or np.nan
        # trans_merchant = any(test_data_trans.loc[test_data_trans.UID == uid, 'merchant'].value_counts()) and test_data_trans.loc[test_data_trans.UID == uid, 'merchant'].value_counts().idxmax() or np.nan
        trans_amt_src1 = any(test_data_trans.loc[test_data_trans.UID == uid, 'amt_src1'].value_counts()) and test_data_trans.loc[test_data_trans.UID == uid, 'amt_src1'].value_counts().idxmax() or np.nan
        # trans_market_code = any(test_data_trans.loc[test_data_trans.UID == uid, 'market_code'].value_counts()) and test_data_trans.loc[test_data_trans.UID == uid, 'market_code'].value_counts().idxmax() or np.nan
        trans_recency = any(test_data_trans.loc[test_data_trans.UID == uid, 'day']) and test_data_trans.loc[test_data_trans.UID == uid, 'day'].max() or 0
        # trans_acc_id1 = any(test_data_trans.loc[test_data_trans.UID == uid, 'acc_id1'].value_counts()) and test_data_trans.loc[test_data_trans.UID == uid, 'acc_id1'].value_counts().idxmax() or np.nan
        # trans_ip1_sub = any(test_data_trans.loc[test_data_trans.UID == uid, 'ip1_sub'].value_counts()) and test_data_trans.loc[test_data_trans.UID == uid, 'ip1_sub'].value_counts().idxmax() or np.nan
        trans_average_money = test_data_trans.loc[test_data_trans.UID == uid, 'trans_amt'].mean()

    test_data.loc[
        (test_data.UID == uid), ['opera_sum', 'opera_sum_not_succe', 'trans_sum', 'trans_all_money', 'trans_channel', 'trans_amt_src1', 'trans_recency']] = \
        [opera_sum, not_succe, trans_sum, trans_all_money, trans_channel, trans_amt_src1, trans_recency]  # 操作的总次数作为一列
# print(test_data[['opera_sum', 'opera_average_time', 'opera_sum_not_succe', 'opera_os', 'trans_sum', 'trans_all_money', 'trans_channel', 'trans_merchant' , 'trans_market_code', 'trans_amt_src1']])
# ————————————————————————预处理处理分割线——————————————————————————————————————————
# train_data = train_data.drop('UID', axis=1)
# test_data = test_data.drop('UID', axis=1)



train_col = train_data.columns




for i in train_col:
    if i != 'Tag':
        train_data[i] = train_data[i].replace('\\N', -1)
        test_data[i] = test_data[i].replace('\\N', -1)

train_data['opera_sum'] = train_data['opera_sum'].astype('int64')  # 1
# train_data['opera_average_time'] = train_data['opera_average_time'].astype('float64')  # 2
train_data['opera_sum_not_succe'] = train_data['opera_sum_not_succe'].astype('float64')  # 3
train_data['trans_sum'] = train_data['trans_sum'].astype('int64')  # 4
train_data['trans_all_money'] = train_data['trans_all_money'].astype('float64')  # 5

test_data['opera_sum'] = test_data['opera_sum'].astype('int64')  # 1
# test_data['opera_average_time'] = test_data['opera_average_time'].astype('float64')  # 2
test_data['opera_sum_not_succe'] = test_data['opera_sum_not_succe'].astype('float64')  # 3
test_data['trans_sum'] = test_data['trans_sum'].astype('int64')  # 4
test_data['trans_all_money'] = test_data['trans_all_money'].astype('float64')  # 5

train_data = train_data.reset_index(drop=True)  # reset_index，通过函数 drop=True 删除原行索引
test_data = test_data.reset_index(drop=True)  # reset_index，通过函数 drop=True 删除原行索引

label2Tag = dict(zip(range(0, len(set(train_data['Tag']))), sorted(list(set(train_data['Tag'])))))
Tag2label = dict(zip(sorted(list(set(train_data['Tag']))), range(0, len(set(train_data['Tag'])))))

Y = train_data['Tag'].map(Tag2label)
train_data = train_data.drop('Tag', axis=1)
test_data = test_data.drop('Tag', axis=1)
train_uid_list = list(train_data['UID'])
test_uid_list = list(test_data['UID'])
onehotpd = pd.concat([train_data, test_data])
onehotpd = onehotpd.reset_index(drop=True)
cols0 = ['trans_amt_src1',]
for i in cols0:
    onehotpd[i][onehotpd[i].isnull()] = 'U0'
    onehotpd[i+'DP'] = onehotpd[i].map(lambda x: re.compile("([a-zA-Z0-9]+)").search(x).group())
    onehotpd[i + 'DP'] = pd.factorize(onehotpd[i+'DP'])[0]
    onehotpd = onehotpd.drop(i, axis=1)

cols = ['trans_channel']

for i in cols:
    ont_hot = pd.get_dummies(onehotpd[i], prefix=i)
    one_hot_DP = pd.DataFrame(ont_hot)
    onehotpd = pd.concat([onehotpd, one_hot_DP], axis=1)
    onehotpd = onehotpd.drop(i, axis=1)
train_data = onehotpd[onehotpd['UID'].isin(train_uid_list)]
test_data = onehotpd[onehotpd['UID'].isin(test_uid_list)]
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

train_data.to_csv('train_data3.csv', index=None)
test_data.to_csv('test_data.csv', index=None)

train_data.drop('UID', axis=1)

train_cols_new = train_data.columns
X = train_data
# print(X)
# print(X.info())
X_test = test_data[train_cols_new]
test_id = test_data['UID']

X, Y, X_test = X.values, Y, X_test.values
# print(X)



def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    score_vali = f1_score(y_true=labels, y_pred=preds, average='weighted')
    # average参数：如果是二分类问题则选择参数‘binary’；如果考虑类别的不平衡性，需要计算类别的加权平均，则使用‘weighted’；如果不考虑类别的不平衡性，计算宏平均，则使用‘macro’。
    return 'f1_score', -score_vali

xx_score=[]

# random_state:随机种子，在shuffle==True时使用，默认使用np.random。
skf = StratifiedKFold(n_splits=n_splites, random_state=seed, shuffle=True)  # n_splits：表示划分几等份shuffle：在每次划分时，是否进行洗牌
for index, (train_index, test_index) in enumerate(skf.split(X, Y)):
    X_train, X_valid, Y_train, Y_valid = X[train_index], X[test_index], Y[train_index], Y[test_index]
    train = xgb.DMatrix(X_train, Y_train)
    validation = xgb.DMatrix(X_valid, Y_valid)
    X_val = xgb.DMatrix(X_valid)
    test = xgb.DMatrix(X_test)
    wachlist = [(validation, 'val')]
    bst = xgb.train(params, train, 10000, wachlist, early_stopping_rounds=15, feval=f1_score_vali, verbose_eval=1)
#     xgb.plot_importance(bst)
    xx_pred = bst.predict(X_val)
#     print(xx_pred.shape)
    xx_score.append(f1_score(Y_valid, xx_pred, average='weighted'))
    Y_test = bst.predict(test)
#     print(Y_test.shape)
    Y_test = Y_test.astype('int64')
    if index == 0:
        cv_pred=np.array(Y_test).reshape(-1, 1)
    else:
        cv_pred=np.hstack((cv_pred, np.array(Y_test).reshape(-1, 1)))


plot_importance(bst)
plt.show()
     

submit = []
for line in cv_pred:
    submit.append(np.argmax(np.bincount(line)))
#     print(np.argmax(np.bincount(line)))


df_test = pd.DataFrame()
df_test['UID'] = list(test_id.unique())
df_test['Tag'] = submit
df_test['Tag'] = df_test['Tag'].map(label2Tag)
df_test.to_csv('result.csv', index=False)
print(xx_score, np.mean(xx_score))


