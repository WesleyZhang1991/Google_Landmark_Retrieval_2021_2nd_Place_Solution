# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import os
import pickle

import os.path as osp

from .bases import BaseImageDataset

class OURAPI(BaseImageDataset):

    def __init__(self, root_train='', root_val='', verbose=True, config=None, **kwargs):
        super(OURAPI, self).__init__()
        self.train_dir = osp.join(root_train, 'trainval')
        self.query_dir = osp.join(root_val, 'test_probe')
        self.gallery_dir = osp.join(root_val, 'test_gallery')
        self.config = config

        # self._check_before_run()

        cache_list = 'cache/' + self.config.DATALOADER.CACHE_LIST
        if not os.path.exists(cache_list):
            train = self._process_dir(self.train_dir, relabel=True)
        else:
            print('Using pre-cached data list')
            train = pickle.load(open(cache_list, 'rb'))
            pid_dic = {}
            print(f'{len(train)} raw images')
            for item in train:
                pid = item[1]
                if pid < 0:
                    continue
                if pid not in pid_dic:
                    pid_dic[pid] = [item]
                else:
                    pid_dic[pid].append(item)
            new_train = []
            relabel_id = 0
            for pid in pid_dic:
                if len(pid_dic[pid]) < self.config.DATALOADER.REMOVE_TAIL:
                    continue
                for item in pid_dic[pid]:
                    # relabel
                    # temp = list(item)
                    # temp[1] = relabel_id
                    # new_train.append(tuple(temp))
                    new_train.append(item)
                relabel_id += 1
            train = new_train
            print(f'{len(train)} images, {relabel_id} valid pids')
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            #print("=> ourapi loaded from: {} and {}".format(root_train,root_val))
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png')) + glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_([\d]+)_([\d]+)')

        # add by gongyou.zyq
        pid_count = {}
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            pid, _, _ = map(int, pattern.search(img_name).groups())
            if pid == -1: continue  # junk images are just ignored
            if pid not in pid_count:
                pid_count[pid] = 1
            else:
                pid_count[pid] += 1
        valid_img_paths = []
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            pid, _, _ = map(int, pattern.search(img_name).groups())
            if pid == -1: continue  # junk images are just ignored
            # NOTE: trick here
            # if pid_count[pid] < self.config.DATALOADER.REMOVE_TAIL:
            # if pid_count[pid] < self.config.DATALOADER.REMOVE_TAIL and pid > 10000000:
            if pid >= 10000000:
                continue
            valid_img_paths.append(img_path)
        if relabel: 
            img_paths = valid_img_paths

        pid_container = set()
        for img_path in sorted(img_paths):
            img_name = os.path.basename(img_path)
            pid, _, _ = map(int, pattern.search(img_name).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            pid, camid, pidx = map(int, pattern.search(img_name).groups())
            if pid == -1: continue  # junk images are just ignored
            #assert 0 <= pid <= 1501  # pid == 0 means background
            #assert 1 <= camid <= 6
            #camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid,-1))

        return dataset
