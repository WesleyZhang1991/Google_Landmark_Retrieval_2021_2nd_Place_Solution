"""Expand official ILR 2021 retrieval validation set."""

import os
import shutil

def main():
    """Main."""

    expand_dir = ('/home/gongyou.zyq/datasets/instance_search/GLDv2/'
                 'search_images/test_gallery_competition_2021')
    index_dir = '/home/gongyou.zyq/datasets/landmark-retrieval-2021/index'
    for device_id in os.listdir(expand_dir):
        old_path = os.path.join(expand_dir, device_id, device_id+'.jpg')
        new_dir = os.path.join(index_dir, device_id[0], device_id[1],
                               device_id[2])
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        shutil.copy(old_path, new_dir)
main()
