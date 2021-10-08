import pickle
import pandas as pd
import numpy as np


def check_online_offline():
    offline = pickle.load(open('/home/gongyou.zyq/video_object_retrieval/tests/features/ILR2021/baseline/delg_reid.pkl', 'rb'))
    online = pd.read_csv('submission.csv')
    for i in range(1129):
        online_query = online['id'][i]
        online_pred = online['images'][i]
        offline_res = offline[online_query]
        print(offline_res)
        fds
        offline_str = ''
        for item in list(offline_res.keys()):
            offline_str += item.split('/')[0] + ' '
        offline_str = offline_str[:-1]
        try:
            assert online_pred == offline_str
        except AssertionError:
            print('find diffrence')
            print(online_pred)
            print(offline_str)

def convert_online2offline():
    online = pd.read_csv('submission.csv')
    online_pkl = {}
    print(len(online['id']))
    for i in range(1129):
        online_query = online['id'][i]
        online_pred = online['images'][i]
        temp_dic = {}
        pred_list = online_pred.split(' ')
        for index, item in enumerate(pred_list):
            temp_dic[f'{item}/{item}.jpg'] = {'bbox': np.array([[0.0, 0.0, 0.0, 0.0]]), 'sim': np.array([1.0 - index/100.0])}
            # temp_dic[f'{item}/{item}.jpg'] = {'bbox': np.array([[0.0, 0.0, 0.0, 0.0]]), 'sim': np.array([1.0 - i/1129.0])}
        online_pkl[online_query] = temp_dic
    pickle.dump(online_pkl, open('submission.pkl', 'wb'))

# check_online_offline()
convert_online2offline()
