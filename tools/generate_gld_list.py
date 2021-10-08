"""Generate cached GLD list. Convert annotation files.

"""
from multiprocessing import Pool
import os
import pickle
import random
from collections import OrderedDict

import numpy as np
import json
import urllib.request
import pandas as pd

RAW_IMAGE_DIR = '/home/gongyou.zyq/datasets/google_landmark/'
CACHE_TRAIN_CLEAN = './cache/cache_clean_list.pkl'
CACHE_TRAIN_FULL = './cache/cache_full_list.pkl'
CACHE_TRAIN_C2X = './cache/cache_c2f_list.pkl'
CACHE_INDEX_TRAIN = './cache/cache_index_train_list.pkl'
CACHE_ALL = './cache/cache_all_list.pkl'
CACHE_TEST = './cache/cache_test_list.pkl'
TRAIN_INFO = './cache/train_info.pkl'
INDEX_INFO = './cache/index_info.pkl'
COUNTRY_INFO = './cache/gbid2country.pkl'


def make_clean_train():
    """Make clean train."""

    train_label = os.path.join(RAW_IMAGE_DIR,
                               'train/train_clean.csv')
    with open(train_label, 'r') as f_train:
        train_data_list = f_train.readlines()[1:]

    tr_landmark2cleangbid = OrderedDict()
    cleangbid2img = OrderedDict()
    image_count = 0
    # NOTE: The order!
    trainclean_dataset = []
    for global_id, line_data in enumerate(train_data_list):
        trainclean_lid = line_data.split(',')[0]
        tr_landmark2cleangbid[trainclean_lid] = global_id
        line_data = line_data.strip()
        image_list = line_data.split(',')[-1].split(' ')
        cleangbid2img[global_id] = image_list
        image_count += len(image_list)
        for image_name in image_list:
            short_dir = f'{image_name[0]}/{image_name[1]}/{image_name[2]}'
            image_path = os.path.join(RAW_IMAGE_DIR, 'train', short_dir,
                                      image_name+'.jpg')
            camid = 0
            trainclean_dataset.append((image_path, global_id, camid, -1))
    pickle.dump(trainclean_dataset, open(CACHE_TRAIN_CLEAN, 'wb'),
                pickle.HIGHEST_PROTOCOL)
    print(f'{image_count} train clean images, '
          f'{len(tr_landmark2cleangbid)} ids')
    return tr_landmark2cleangbid, cleangbid2img


def make_full_train(tr_landmark2cleangbid, cleangbid2img):
    """Make full trian."""

    train_label = os.path.join(RAW_IMAGE_DIR, 'train/train.csv')
    with open(train_label, 'r') as f_train:
        train_data_list = f_train.readlines()[1:]
    trainfull_dataset = []
    trainfull_expand_clean = []
    tr_img2landmark = OrderedDict()
    tr_landmark2fullgbid = OrderedDict()
    gid_offset = len(tr_landmark2cleangbid)
    add_lid_count = {}
    for line_data in train_data_list:
        line_data = line_data.strip()
        image_name, _, landmark_id = line_data.split(',')
        tr_img2landmark[image_name] = landmark_id
        short_dir = f'{image_name[0]}/{image_name[1]}/{image_name[2]}'
        image_path = os.path.join(RAW_IMAGE_DIR, 'train', short_dir,
                                  image_name+'.jpg')
        # NOTE: we use _01_ camid to sepate noisy data
        camid = 1
        if landmark_id in tr_landmark2cleangbid:
            global_id = tr_landmark2cleangbid[landmark_id]
            if image_name in cleangbid2img[global_id]:
                camid = 0
            trainfull_expand_clean.append((image_path, global_id, camid, -1))
        else:
            if landmark_id not in add_lid_count:
                add_lid_count[landmark_id] = len(add_lid_count)
            global_id = gid_offset + add_lid_count[landmark_id]
        tr_landmark2fullgbid[landmark_id] = global_id
        trainfull_dataset.append((image_path, global_id, camid, -1))
    pickle.dump(trainfull_dataset, open(CACHE_TRAIN_FULL, 'wb'),
                pickle.HIGHEST_PROTOCOL)
    pickle.dump(trainfull_expand_clean, open(CACHE_TRAIN_C2X, 'wb'),
                pickle.HIGHEST_PROTOCOL)
    assert len(tr_landmark2fullgbid) == len(tr_landmark2cleangbid) + \
                                    len(add_lid_count)
    print(f'{len(trainfull_dataset)} train full images, '
          f'{len(trainfull_expand_clean)} trainfull expand clean images, '
          f'{len(tr_landmark2fullgbid)} trainfull ids')
    return tr_img2landmark, tr_landmark2fullgbid


def load_category(category_file):
    """Load category."""

    label_to_category = os.path.join(RAW_IMAGE_DIR, category_file)
    with open(label_to_category, 'r') as f_cat:
        category_data_list = f_cat.readlines()[1:]
    lid2cat = OrderedDict()
    cat2lid = OrderedDict()
    for line_data in category_data_list:
        line_data = line_data.strip()
        # pylint: disable=line-too-long
        if '\"http://' in line_data:
            prefix_str = ',\"http://commons.wikimedia.org/wiki/Category:'
            landmark_id, category_name = line_data.split(prefix_str)
            category_name = category_name[:-1]
        else:
            prefix_str = ',http://commons.wikimedia.org/wiki/Category:'
            landmark_id, category_name = line_data.split(prefix_str)
        # if 'Marb%C3%A4cks_kyrka,_Sm%C3%A5land' in category_name:
        #     print(line_data)
        lid2cat[landmark_id] = category_name
        cat2lid[category_name] = landmark_id
    return lid2cat, cat2lid


def make_train():
    """Make train."""

    tr_landmark2cleangbid, cleangbid2img = make_clean_train()
    tr_img2landmark, tr_landmark2fullgbid = make_full_train(
            tr_landmark2cleangbid, cleangbid2img)
    train_category = 'train/train_label_to_category.csv'
    tr_landmark2category, tr_category2landmark = load_category(train_category)
    _ = tr_category2landmark
    pickle.dump({'tr_img2landmark': tr_img2landmark,
                 'tr_landmark2fullgbid': tr_landmark2fullgbid,
                 'tr_landmark2cleangbid': tr_landmark2cleangbid,
                 'tr_landmark2category': tr_landmark2category},
                open(TRAIN_INFO, 'wb'), pickle.HIGHEST_PROTOCOL)


def make_index():
    """Make index."""

    idx_img2category = OrderedDict()
    index_category = 'index/index_label_to_category.csv'
    idx_landmark2category, idx_category2landmark = load_category(
            index_category)
    index_image_to_landmark = os.path.join(RAW_IMAGE_DIR,
                                           'index/index_image_to_landmark.csv')
    with open(index_image_to_landmark, 'r') as f_index:
        index_data_list = f_index.readlines()[1:]
    for index_data in index_data_list:
        image_name, idx_landmark = index_data.strip().split(',')
        category = idx_landmark2category[idx_landmark]
        idx_img2category[image_name] = category
    pickle.dump({'idx_img2category': idx_img2category,
                 'idx_category2landmark': idx_category2landmark},
                open(INDEX_INFO, 'wb'), pickle.HIGHEST_PROTOCOL)


def simple_reverse(in_dict):
    """Simple reverse dict key and value."""

    out_dict = OrderedDict()
    for k, v in in_dict.items():
        out_dict[v] = k
    return out_dict


def merge_train_index():
    """Merge trian index."""

    train_info = pickle.load(open(TRAIN_INFO, 'rb'))
    index_info = pickle.load(open(INDEX_INFO, 'rb'))
    tr_img2landmark = train_info['tr_img2landmark']
    tr_landmark2category = train_info['tr_landmark2category']
    tr_category2landmark = simple_reverse(tr_landmark2category)
    tr_landmark2fullgbid = train_info['tr_landmark2fullgbid']
    # idx_category2landmark = index_info['idx_category2landmark']
    tr_img2category = OrderedDict((k, tr_landmark2category[v])
                                   for k, v in tr_img2landmark.items())
    tr_category2gbid = OrderedDict((k, tr_landmark2fullgbid[v])
                                    for k, v in tr_category2landmark.items())
    merge_img2category = tr_img2category
    merge_category2gbid = tr_category2gbid
    # merge_img2gbid = OrderedDict()

    # add_lid_count = {}
    gbid_offset = len(train_info['tr_landmark2fullgbid'])
    print(f'{gbid_offset} gbid_offset')
    index_trainfull_dataset = pickle.load(open(CACHE_TRAIN_FULL, 'rb'))
    test_dataset = []
    test_id_count = {}
    for image_name, idx_category in index_info['idx_img2category'].items():
        short_dir = f'{image_name[0]}/{image_name[1]}/{image_name[2]}'
        image_path = os.path.join(RAW_IMAGE_DIR, 'index', short_dir,
                                  image_name+'.jpg')
        # NOTE, index lid may not include in trainfull ids, just ignore these.
        """
        # Also include index landmark categories that are not in trainfull.
        if idx_category not in merge_category2gbid:
            if idx_category not in add_lid_count:
                add_lid_count[idx_category] = len(add_lid_count)
            global_id = gbid_offset + add_lid_count[idx_category]
            merge_category2gbid[idx_category] = global_id
        else:
            global_id = merge_category2gbid[idx_category]
        """
        if idx_category not in merge_category2gbid:
            continue
        # NOTE: fix by 20210929
        merge_img2category[image_name] = idx_category
        global_id = merge_category2gbid[idx_category]
        camid = 0
        index_trainfull_dataset.append((image_path, global_id, camid, -1))
        if global_id not in test_id_count:
            test_id_count[global_id] = 1
        else:
            test_id_count[global_id] += 1
        if test_id_count[global_id] < 3:
            test_dataset.append((image_path, global_id, camid, -1))
    pickle.dump(index_trainfull_dataset, open(CACHE_INDEX_TRAIN, 'wb'),
                pickle.HIGHEST_PROTOCOL)
    test_dataset = random.sample(test_dataset, 20000)
    pickle.dump(test_dataset, open(CACHE_TEST, 'wb'),
                pickle.HIGHEST_PROTOCOL)
    print(f'{len(index_trainfull_dataset)} index_trainfull images, '
          f'{len(merge_category2gbid)} index_trainfull ids/categories')
    print(f'{len(test_dataset)} test images, {len(test_id_count)} test ids')
    return merge_img2category, merge_category2gbid


def complex_reverse(in_dict):
    """Complex reverse."""

    tmp_values = set(in_dict.values())
    out_dict = {v: [] for v in tmp_values}
    # pylint: disable=invalid-name
    for k, v in in_dict.items():
        out_dict[v].append(k)
    return out_dict


def merge_test_index_train(merge_img2category, merge_category2gbid):
    """Merge test index train."""

    test_gt = os.path.join(RAW_IMAGE_DIR,
                           'test/retrieval_solution_v2.1.csv')
    with open(INDEX_INFO, 'rb') as f_index_info:
        index_info = pickle.load(f_index_info)
    idx_img2category = index_info['idx_img2category']
    with open(test_gt, 'r') as f_gt:
        line_data_list = f_gt.readlines()[1:]

    test_index_trainfull_dataset = pickle.load(open(CACHE_INDEX_TRAIN, 'rb'))
    index_trainfull_num = len(test_index_trainfull_dataset)
    print(f'{index_trainfull_num} index_trainfull images before merge test')
    refindex2gbid = {}
    for i in range(index_trainfull_num):
        refindex2gbid[i] = test_index_trainfull_dataset[i][1]
    for line_data in line_data_list:
        probe_name, index_name_str, subset = line_data.strip().split(',')
        short_dir = f'{probe_name[0]}/{probe_name[1]}/{probe_name[2]}'
        image_path = os.path.join(RAW_IMAGE_DIR, 'test', short_dir,
                                  probe_name+'.jpg')
        camid = 0
        if subset == 'Ignored':
            # add non-landmark, and give -1 id
            global_id = -1
            test_index_trainfull_dataset.append((image_path, global_id, camid, -1))
            continue
        if "\"" in index_name_str:
            index_name_str = index_name_str[1:-1]

        index_gbid_list = []
        for image_name in index_name_str.split(' '):
            tmp_category = idx_img2category[image_name]
            # there might beyond 203094 ids
            if tmp_category not in merge_category2gbid:
                continue
            index_gbid = merge_category2gbid[tmp_category]
            index_gbid_list.append(index_gbid)
        unique_gbid_list = list(set(index_gbid_list))

        """
        # add the test image itself
        global_id = unique_gbid_list[0]
        test_index_trainfull_dataset.append((image_path, global_id, camid, -1))
        # images list in gt must be in the index gt, same number!
        if len(unique_gbid_list) > 1:
            # change id for re-annotated data.
            # print(unique_gbid_list)
            for gbid in unique_gbid_list:
                refindexes = gbid2refindexes[gbid]
                for refindex in refindexes:
                    old_tuple = test_index_trainfull_dataset[refindex]
                    old_list = list(old_tuple)
                    old_list[1] = global_id
                    test_index_trainfull_dataset[refindex] = tuple(old_list)
        """
        # same query image can be given multiple tags
        for gbid in unique_gbid_list:
            test_index_trainfull_dataset.append((image_path, gbid, camid, -1))
    print(f'{len(test_index_trainfull_dataset)} all images in 203094 classes')
    print('Plus one non-landmark class -1 label')
    pickle.dump(test_index_trainfull_dataset, open(CACHE_ALL, 'wb'),
                pickle.HIGHEST_PROTOCOL)


def get_country_info():
    """Get country info."""

    download_flag = True
    country_json = './cache/country_info/country_codes.json'
    country_codes = json.load(open(country_json))
    if download_flag:
        for country_name in country_codes:
            url = f'https://storage.googleapis.com/gld-v2/data/train/country/{country_name}.json'
            urllib.request.urlretrieve(url, f"./cache/country_info/{country_name}.json")
        print('Finish downloading country info')
    category2country = {}
    countryname2countryid = {}
    count = 0
    for country_name in country_codes:
        countryname2countryid[country_name] = count
        count += 1
        country_category = json.load(open(f"./cache/country_info/{country_name}.json"))
        for item in country_category:
            cat_name = item['name']
            category2country[cat_name] = country_name
    train_info = pickle.load(open(TRAIN_INFO, 'rb'))
    gbid2countryid = OrderedDict()
    gbid2countryname = OrderedDict()
    tr_landmark2fullgbid = train_info['tr_landmark2fullgbid']
    tr_landmark2category = train_info['tr_landmark2category']
    gbid2landmark = simple_reverse(tr_landmark2fullgbid)
    for gbid, landmark_id in gbid2landmark.items():
        cat_name = tr_landmark2category[landmark_id]
        country_name = category2country[cat_name]
        country_id = countryname2countryid[country_name]
        gbid2countryid[gbid] = country_id
        gbid2countryname[gbid] = country_name
    print(f'{len(category2country)} categories with country info in total')
    pickle.dump(gbid2countryid, open(COUNTRY_INFO, 'wb'),
                pickle.HIGHEST_PROTOCOL)
    return gbid2countryname


def refine_all_dataset(gbid2countryname):
    """Refine all dataset."""

    df = pd.read_csv('cache/country_info/continent_codes.csv')
    continent_name = np.array(df['Continent_Name'])
    two_letter_code = np.array(df['Two_Letter_Country_Code'])
    country2continent = {}
    for (x, y) in zip(continent_name, two_letter_code):
        country2continent[y] = x

    all_dataset = pickle.load(open(CACHE_ALL, 'rb'))
    new_dataset = []
    for item in all_dataset:
        gbid = item[1]
        if gbid == -1:
            country = 'null'
            continent = 'null'
        else:
            country = gbid2countryname[gbid]
            if country == 'OTHER':
                continent = 'OTHER'
            elif country == 'XK':
                continent = 'Europe'
            else:
                continent = country2continent[country]
        item = list(item)
        item[-1] = continent
        new_dataset.append(tuple(item))
    pickle.dump(new_dataset, open(CACHE_ALL, 'wb'),
                pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    make_train()
    make_index()
    # NOTE: test have no landmark_id/category, so ignored.
    merge_img2category, merge_category2gbid = merge_train_index()
    merge_test_index_train(merge_img2category, merge_category2gbid)

    gbid2countryname = get_country_info()
    refine_all_dataset(gbid2countryname)
