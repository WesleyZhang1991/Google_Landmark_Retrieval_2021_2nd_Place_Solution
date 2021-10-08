"""Base code for ILR2021"""

import gc
import gzip
import os
import pickle
import shutil
import sys
import time

import cv2
import psutil
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

sys.path.append('../input/util-code/')
from make_model import make_model

from landmark_retrieval import setting, write_csv, category_rerank_after_merge, category_rerank_before_merge, get_probe_tags,get_index_tags, merge_tags, get_probe_tags_index, get_probe_tags_topk

# pylint: disable=not-callable, invalid-name, line-too-long
# ########################## Cell 1 Basic module test ########################

try:
    assert torch.cuda.is_available()
    gpu_num = torch.cuda.device_count()
    assert gpu_num > 0
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    print(f'Total {gpu_num} gpu cards with {gpu_memory} memory')
except AssertionError:
    print('Fail to set gpu')

# ########################## Cell 3 Load all image list  #####################


def load_image_list():
    """Load image list."""

    query_count = 0
    index_count = 0
    all_image_list = []
    for dirname, _, filenames in os.walk(setting['PROBE_DIR']):
        for filename in filenames:
            query_count += 1
            all_image_list.append(os.path.join(dirname, filename))
            # print(os.path.join(dirname, filename))
    if query_count == setting['SAMPLE_TEST_NUM'] and gpu_num == 1:
        return None, None
    for dirname, _, filenames in os.walk(setting['INDEX_DIR']):
        for filename in filenames:
            index_count += 1
            all_image_list.append(os.path.join(dirname, filename))
            # print(os.path.join(dirname, filename))
    print(f'query num: {query_count} and index num: {index_count}')
    return all_image_list, query_count

# ########################## Cell 4 ReID inference  ##########################


class ImageDataset(Dataset):
    """Image Dataset."""

    def __init__(self, dataset, transforms):
        _ = transforms
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (setting['IMAGE_SIZE'], setting['IMAGE_SIZE']))
        img = torch.tensor(img)
        img = img[:, :, [2, 1, 0]]
        return img, img_path


def val_collate_fn(batch):
    """Val collate fn."""

    imgs, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), img_paths


class ReID_Inference:
    """ReID Inference."""

    def __init__(self, backbone):

        self.model = make_model(setting['MODEL_PARAMS'][backbone]['BACKBONE'])
        model_name = setting['MODEL_PARAMS'][backbone]['MODEL_NAME']
        model_path = os.path.join(setting['MODEL_DIR'], model_name)
        self.model.load_param(model_path)
        self.batch_size = setting['BATCH_SIZE']
        if gpu_num > 1:
            print(f'Using {gpu_num} gpu for inference')
            self.model = nn.DataParallel(self.model)
            self.batch_size = setting['BATCH_SIZE'] * gpu_num
        self.model.to('cuda')
        self.model.eval()
        self.mean = torch.tensor([123.675, 116.280, 103.530]).to('cuda')
        self.std = torch.tensor([57.0, 57.0, 57.0]).to('cuda')

    def extract(self, imgpath_list):
        """Extract feature for one image."""

        val_set = ImageDataset(imgpath_list, None)

        # NOTE: no pin_memory to save memory
        if gpu_num > 1:
            pin_memory = True
            num_workers = 32
        else:
            pin_memory = False
            num_workers = 2
        val_loader = DataLoader(
            val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=val_collate_fn,
            pin_memory=pin_memory
        )

        batch_res_dic = OrderedDict()
        for (batch_data, batch_path) in tqdm(val_loader,
                                             total=len(val_loader)):
            with torch.no_grad():
                batch_data = batch_data.to('cuda')
                batch_data = (batch_data - self.mean) / self.std
                batch_data = batch_data.permute(0, 3, 1, 2)
                batch_data = batch_data.float()
                if not setting['MULTI_SCALE_FEAT']:
                    if setting['FP16']:
                        # NOTE: DO NOT use model.half() because of underflow
                        with amp.autocast():
                            feat = self.model(batch_data)
                    else:
                        feat = self.model(batch_data)
                else:
                    # Ref: https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/3fee857dd2b2927ede70c43bffd99b41eb394507/cirtorch/networks/imageretrievalnet.py#L309
                    feat = torch.zeros((len(batch_data), 512),
                                       dtype=torch.float16).cuda()
                    raw_size = batch_data.shape[2]
                    for s in [0.707, 1.0, 1.414]:
                        new_size = int(((raw_size * s) // 16) * 16)
                        scale_data = nn.functional.interpolate(
                                batch_data, size=new_size, mode='bilinear',
                                align_corners=False)
                        with amp.autocast():
                            scale_feat = self.model(scale_data)
                        feat += scale_feat
                    feat = feat/3.0
                feat = feat / torch.norm(feat, 2, 1, keepdim=True)
                feat = feat.cpu().detach().numpy()

            for index, imgpath in enumerate(batch_path):
                batch_res_dic[imgpath] = feat[index]
        del val_loader, val_set, feat, batch_data
        return batch_res_dic


def debug_reid_inference(image_list):
    """Debug reid inference."""

    reid = ReID_Inference('R50')
    batch_res_dic = reid.extract(image_list[:20])
    print(batch_res_dic)
    del reid, batch_res_dic

# ########################## Cell 5 Extract feature  #########################


def save_feature(all_feature_dic, backbone):
    """Save feature."""

    if not os.path.exists(setting['FEAT_DIR']):
        os.makedirs(setting['FEAT_DIR'])
    index_name_list, index_feats = [], []
    probe_name_list, probe_feats = [], []
    # NOTE: attention the order! Related to probe_name_list order
    for image_path, sample_feat in sorted(all_feature_dic.items()):
        image_name = os.path.basename(image_path).split('.jpg')[0]
        sample_mode = image_path.split('/')[-5]
        if sample_mode == 'test':
            probe_name_list.append(image_name)
            probe_feats.append(sample_feat)
        else:
            index_name_list.append(image_name)
            index_feats.append(sample_feat)

    pkl_name = os.path.join(setting['FEAT_DIR'], f'probe_feats_{backbone}.pkl')
    probe_dic = {'probe_name_list': np.array(probe_name_list),
                 'probe_feats': np.array(probe_feats)}
    with open(pkl_name, 'wb') as f_pkl:
        pickle.dump(probe_dic, f_pkl, pickle.HIGHEST_PROTOCOL)
    print('Save pickle in %s' % pkl_name)
    pkl_name = os.path.join(setting['FEAT_DIR'], f'index_feats_{backbone}.pkl')
    index_dic = {'index_name_list': np.array(index_name_list),
                 'index_feats': np.array(index_feats)}
    with open(pkl_name, 'wb') as f_pkl:
        pickle.dump(index_dic, f_pkl, pickle.HIGHEST_PROTOCOL)
    print('Save pickle in %s' % pkl_name)
    all_feature_dic.clear(), probe_dic.clear(), index_dic.clear()
    del all_feature_dic, probe_dic, index_dic, probe_feats, index_feats
    del probe_name_list, index_name_list
    gc.collect()


def load_feat(mode, backbone):
    """Load precomputed features."""

    feat_dir = setting['FEAT_DIR']
    with open(f'{feat_dir}/{mode}_feats_{backbone}.pkl', 'rb') as f_pkl:
        mode_dic = pickle.load(f_pkl)
    print(f'load {backbone} feat, memory : {psutil.virtual_memory().percent}')
    return mode_dic[f'{mode}_name_list'], mode_dic[f'{mode}_feats']


def save_numpy(data_path, data, save_disk_flag=True):
    """Save numpy."""

    if save_disk_flag:
        # Save space but slow
        f_data = gzip.GzipFile(f"{data_path}.gz", "w")
        np.save(file=f_data, arr=data)
        f_data.close()
    else:
        np.save(data_path, data)


def load_numpy(data_path, save_disk_flag=True):
    """Load numpy."""

    if save_disk_flag:
        # Save space but slow
        f_data = gzip.GzipFile(f'{data_path}.gz', "r")
        data = np.load(f_data)
    else:
        data = np.load(data_path)
    return data


# ########################## Cell 6 KR rerank sims  #########################


def build_graph(initial_rank):
    """Build graph."""

    K1 = setting['K1']
    if not os.path.exists(setting['GRAPH_DIST_DIR']):
        os.makedirs(setting['GRAPH_DIST_DIR'])

    torch.cuda.empty_cache()  # empty GPU memory
    gc.collect()
    print(f'Start build graph, memory: {psutil.virtual_memory().percent}')
    all_num = initial_rank.shape[0]
    for i in tqdm(range(all_num)):
        original_dist = load_numpy(os.path.join(setting['EUC_DIST_DIR'],
                                                f'{i:08d}.npy'),
                                   save_disk_flag=False)
        V = np.zeros_like(original_dist, dtype=np.float16)
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :K1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :K1+1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(K1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index, :int(np.around(K1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[k_reciprocal_expansion_index])
        norm_weight = weight/np.sum(weight)
        V[k_reciprocal_expansion_index] = norm_weight
        save_numpy(os.path.join(setting['GRAPH_DIST_DIR'], f'{i:08d}.npy'), V)
    print(f'Finish build graph, memory: {psutil.virtual_memory().percent}')


def expand_query(initial_rank):
    """Expand query."""

    K2 = setting['K2']
    print(f'Start QE, memory usage: {psutil.virtual_memory().percent}')
    if not os.path.exists(setting['QE_DIST_DIR']):
        os.makedirs(setting['QE_DIST_DIR'])

    all_num = len(initial_rank)
    for i in tqdm(range(all_num)):
        query_neighbor_list = initial_rank[i, :K2]
        neighbor_dist_list = []
        for j in query_neighbor_list:
            neighbor_file = os.path.join(setting['GRAPH_DIST_DIR'],
                                         f'{j:08d}.npy')
            neighbor_dist = load_numpy(neighbor_file)
            neighbor_dist_list.append(neighbor_dist)
        neighbor_dist_list = np.array(neighbor_dist_list)
        mean_dist = np.mean(neighbor_dist_list, axis=0)
        save_numpy(os.path.join(setting['QE_DIST_DIR'], f'{i:08d}.npy'),
                   mean_dist)
    print(f'Finish QE, memory usage: {psutil.virtual_memory().percent}')


def compute_jaccard(query_num, all_num):
    """Compute Jaccard distance."""

    JACCARD_DIR = setting['JACCARD_DIR']
    QE_DIST_DIR = setting['QE_DIST_DIR']
    if not os.path.exists(JACCARD_DIR):
        os.makedirs(JACCARD_DIR)

    gc.collect()
    print(f'Start Jaccard, memory usage: {psutil.virtual_memory().percent}')

    gal_nonzero_dic = {k: [] for k in range(all_num)}
    prb_nonzero_dic = {k: [] for k in range(all_num)}
    for k in range(all_num):
        sample_dist = load_numpy(os.path.join(QE_DIST_DIR, f'{k:08d}.npy'))
        indexes = np.where(sample_dist != 0)[0]
        for gal in indexes:
            if gal in gal_nonzero_dic:
                gal_nonzero_dic[gal].append(k)
        prb_nonzero_dic[k] = list(indexes)

    invIndex = []
    for i in range(all_num):
        invIndex.append(gal_nonzero_dic[i])
    for i in tqdm(range(query_num)):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float16)
        indNonZero = prb_nonzero_dic[i]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_indNonZero_dist = load_numpy(os.path.join(QE_DIST_DIR,
                                                           f'{i:08d}.npy'))
            temp_indNonZero_dist = temp_indNonZero_dist[indNonZero[j]]
            temp_indImages = indImages[j]
            min_dist_list = []
            for ind in temp_indImages:
                temp_ind_dist = load_numpy(os.path.join(QE_DIST_DIR,
                                                        f'{ind:08d}.npy'))
                min_dist_list.append(temp_ind_dist[indNonZero[j]])
            min_dist_list = np.array(min_dist_list)
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + \
                    np.minimum(temp_indNonZero_dist, min_dist_list)
        jaccard_dist = 1-temp_min/(2-temp_min)
        jaccard_dist = jaccard_dist.flatten()
        save_numpy(os.path.join(JACCARD_DIR, f'{i:08d}.npy'), jaccard_dist)
    print(f'Finish Jaccard, memory usage: {psutil.virtual_memory().percent}')


def merge_sims(query_num, all_num):
    """Merge original dist and jaccard dist."""

    print(f'Start merge sim, memory usage: {psutil.virtual_memory().percent}')
    EUC_DIST_DIR = setting['EUC_DIST_DIR']
    JACCARD_DIR = setting['JACCARD_DIR']
    LAMBDA = setting['LAMBDA']

    index_num = all_num - query_num
    merged_dist = np.zeros((query_num, index_num), dtype=np.float16)
    for i in range(query_num):
        original_dist = load_numpy(os.path.join(EUC_DIST_DIR, f'{i:08d}.npy'),
                                   save_disk_flag=False)
        jaccard_dist = load_numpy(os.path.join(JACCARD_DIR, f'{i:08d}.npy'))
        dist = jaccard_dist*(1-LAMBDA) + original_dist*LAMBDA
        merged_dist[i] = dist[query_num:]
    print(f'Finish merge sim, memory usage: {psutil.virtual_memory().percent}')
    return 1.0 - merged_dist


def get_origin_sims(query_num, all_num):
    """Get origin sims."""

    print(f'Start original, memory usage: {psutil.virtual_memory().percent}')
    EUC_DIST_DIR = setting['EUC_DIST_DIR']

    index_num = all_num - query_num
    merged_dist = np.zeros((query_num, index_num), dtype=np.float16)
    for i in range(query_num):
        original_dist = load_numpy(os.path.join(EUC_DIST_DIR, f'{i:08d}.npy'),
                                   save_disk_flag=False)
        merged_dist[i] = original_dist[query_num:]
    print(f'Finish original, memory usage: {psutil.virtual_memory().percent}')
    return 1.0 - merged_dist


def cache_expand_sims(probe_feats, index_feats):
    """Cache expanded(query + index) sims for KR rerank."""

    if not os.path.exists(setting['EUC_DIST_DIR']):
        os.makedirs(setting['EUC_DIST_DIR'])

    query_num = probe_feats.shape[0]
    index_num = index_feats.shape[0]
    all_num = query_num + index_num
    initial_rank = np.zeros((all_num, setting['K1']+10), dtype=np.int32)
    concat_feat = torch.cat([probe_feats, index_feats])
    del probe_feats, index_feats
    torch.cuda.empty_cache()  # empty GPU memory
    gc.collect()
    print(f'Load feats memory usage: {psutil.virtual_memory().percent}')
    for sample_index in tqdm(range(all_num)):
        cos_sim = torch.matmul(concat_feat[sample_index][None, :],
                               concat_feat.T)
        # euc_dist_gpu = 2 * (1 - cos_sim)
        # euc_dist_gpu = torch.sqrt(euc_dist_gpu)
        # custom euc dist without norm
        euc_dist_gpu = 1.0 - (cos_sim + 1.0)/2
        euc_dist_gpu = euc_dist_gpu[0]
        euc_dist_cpu = euc_dist_gpu.cpu().numpy()
        # print(euc_dist_cpu.shape, euc_dist_cpu.max(), euc_dist_cpu.min())
        orders = torch.argsort(euc_dist_gpu)
        orders = orders.cpu().numpy()[:setting['K1']+10]
        initial_rank[sample_index, :] = orders
        save_numpy(os.path.join(setting['EUC_DIST_DIR'],
                                f'{sample_index:08d}.npy'),
                   euc_dist_cpu, save_disk_flag=False)
        del cos_sim, euc_dist_gpu, euc_dist_cpu, orders
        # print(f'Memory usage: {psutil.virtual_memory().percent}')
    # print(initial_rank.shape)
    save_numpy(setting['INITIAL_RANK_FILE'], initial_rank)
    del concat_feat
    torch.cuda.empty_cache()  # empty GPU memory
    gc.collect()
    print(f'Finish cache sim, memory usage: {psutil.virtual_memory().percent}')
    return initial_rank


def kr_rerank_disk(probe_feats, index_feats):
    """Memory efficient rerank."""

    print('Starting re_ranking')
    initial_rank = cache_expand_sims(probe_feats, index_feats)
    query_num = len(probe_feats)
    all_num = len(initial_rank)
    build_graph(initial_rank)
    if setting['K2'] != 1:
        expand_query(initial_rank)
    print(f'Memory usage: {psutil.virtual_memory().percent}')
    del initial_rank
    gc.collect()
    print(f'Memory usage: {psutil.virtual_memory().percent}')
    compute_jaccard(query_num, all_num)
    sims = merge_sims(query_num, all_num)
    return sims


# ########################## Cell 7 Refset extraction  #######################


def load_meta(backbone):
    """Load meta."""

    with open(setting['REF_SET_META'], 'rb') as f_meta:
        ref_meta = pickle.load(f_meta)
    print(f'Load ref_set_meta, memory: {psutil.virtual_memory().percent}')
    pkl_name = setting['REF_SET_FEAT'].replace('.pkl', f'_{backbone}.pkl')
    refset_feat_dic = pickle.load(open(pkl_name, 'rb'))
    print(f'Load raw ref_set_feat, memory: {psutil.virtual_memory().percent}')
    ref_feats_gpu = []
    ref_feats = []
    batch_size = 128
    for ref_name in ref_meta['ref_name_list']:
        ref_feats.append(refset_feat_dic[ref_name])
        if len(ref_feats) % batch_size == 0:
            ref_feats = np.array(ref_feats)
            ref_feats = torch.tensor(ref_feats).cuda().half()
            ref_feats_gpu.append(ref_feats)
            ref_feats = []
        del refset_feat_dic[ref_name]
    if len(ref_feats) > 0:
        ref_feats = np.array(ref_feats)
        ref_feats = torch.tensor(ref_feats).cuda().half()
        ref_feats_gpu.append(ref_feats)
    del refset_feat_dic
    del ref_meta['ref_name_list']
    ref_meta['ref_feats'] = torch.cat(ref_feats_gpu, dim=0)
    print(f'Convert ref_set_feat, memory: {psutil.virtual_memory().percent}')

    with open(setting['REF_LOC_MAP'], 'rb') as f:
        loc_map = pickle.load(f)
    ref_gbid = []
    ref_loc = []
    refindex2globalid = ref_meta['refindex2globalid']
    for refindex in refindex2globalid:
        gbid = refindex2globalid[refindex]
        ref_gbid.append(gbid)
        loc = loc_map[gbid]
        ref_loc.append(loc)
    ref_gbid = torch.tensor(ref_gbid).cuda()
    ref_loc = torch.tensor(ref_loc).cuda()
    ref_meta['ref_gbid'] = ref_gbid
    ref_meta['ref_loc'] = ref_loc
    print(f'Convert other metas, memory: {psutil.virtual_memory().percent}')
    return ref_meta


def prepare_meta():
    """Prepare meta."""

    # NOTE: The order for ref_path_list, global_id serve as category name
    print('Using %s as ref set' % setting['REF_SET_LIST'])
    with open(setting['REF_SET_LIST'], 'rb') as f_ref:
        ref_set_list = pickle.load(f_ref)
    refindex2globalid, globalid2refindex = OrderedDict(), OrderedDict()
    ref_name_list = []
    global_count = 0
    for item in ref_set_list:
        ref_path = item[0]
        image_name = os.path.basename(ref_path).split('.jpg')[0]
        global_id = item[1]
        # Ignore non-landmarks
        if global_id < 0:
            continue
        assert global_id < 203094
        ref_name_list.append(image_name)
        if global_id not in globalid2refindex:
            globalid2refindex[global_id] = [global_count]
        else:
            globalid2refindex[global_id].append(global_count)
        refindex2globalid[global_count] = global_id
        global_count += 1
    print(f'{len(globalid2refindex)} unique global ids')
    print(f'{global_count} ref images')
    save_dic = {'globalid2refindex': globalid2refindex,
                'refindex2globalid': refindex2globalid,
                'ref_name_list': ref_name_list}
    dirname = os.path.dirname(setting['REF_SET_META'])
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(setting['REF_SET_META'], 'wb') as f_meta:
        pickle.dump(save_dic, f_meta, pickle.HIGHEST_PROTOCOL)


def extract_refset(reid, backbone):
    """Extract feature for ref set."""

    with open(setting['REF_ALL_LIST'], 'rb') as f_ref:
        ref_set_list = pickle.load(f_ref)
    ref_path_list = []
    for item in ref_set_list:
        ref_path = item[0]
        ref_path_list.append(ref_path)

    start_time = time.time()
    all_feature_dic = reid.extract(ref_path_list)
    save_dic = {}
    for ref_path, feat in all_feature_dic.items():
        ref_name = os.path.basename(ref_path).split('.jpg')[0]
        save_dic[ref_name] = feat
    print('%.4f s' % (time.time() - start_time))
    pkl_name = setting['REF_SET_FEAT'].replace('.pkl', f'_{backbone}.pkl')
    with open(pkl_name, 'wb') as f_pkl:
        pickle.dump(save_dic, f_pkl, pickle.HIGHEST_PROTOCOL)
    print(f'Extract refset Memory: {psutil.virtual_memory().percent}')


# ########################## Cell 8 Get output file  #########################


def compute_sim(backbone):
    """Compute initial similarities and ranklist."""

    torch.cuda.empty_cache()  # empty GPU memory
    gc.collect()
    print(f'Start cache sim, memory usage: {psutil.virtual_memory().percent}')
    if not os.path.exists(setting['SIMS_DIR']):
        os.makedirs(setting['SIMS_DIR'])

    probe_name_list, probe_feats = load_feat('probe', backbone)
    index_name_list, index_feats = load_feat('index', backbone)
    print(len(probe_name_list), len(index_name_list))

    probe_feats = torch.tensor(probe_feats).cuda().half()
    index_feats = torch.tensor(index_feats).cuda().half()
    if not setting['KR_FLAG'] and setting['CATEGORY_RERANK'] != 'before_merge':
        sims = torch.matmul(probe_feats, index_feats.T).cpu().numpy()
    elif setting['CATEGORY_RERANK'] == 'before_merge':
        ref_info = load_meta(backbone)
        ref_info['probe_name_list'] = probe_name_list
        ref_info['index_name_list'] = index_name_list

        sims = torch.matmul(probe_feats, index_feats.T)
        index_tags, index_locs = get_index_tags(index_feats, ref_info, batch_size=128)
        probe_tags, probe_tag_scores, probe_locs, probe_loc_scores = get_probe_tags_topk(probe_feats, ref_info)
        sims = category_rerank_after_merge(sims, probe_tags, probe_locs,
                                               probe_tag_scores, probe_loc_scores,
                                               index_tags, index_locs,
                                               sim_thr=setting['CATEGORY_THR'],
                                               alpha=setting['alpha'],
                                               beta=setting['beta'])

    elif setting['KR_FLAG']:
        # NOTE: KR rerank seems not suitable here.
        sims = kr_rerank_disk(probe_feats, index_feats)
    else:
        print('Unkown compute sims setting')
    if torch.is_tensor(sims):
        sims = sims.cpu().numpy()
    pkl_name = os.path.join(setting['SIMS_DIR'], f'{backbone}_sims.pkl')
    with open(pkl_name, 'wb') as f_sims:
        pickle.dump(sims, f_sims, pickle.HIGHEST_PROTOCOL)
    # NOTE: It is important to fix this order for all models.
    with open(setting['NAME_LIST_FILE'], 'wb') as f_name:
        pickle.dump([probe_name_list, index_name_list], f_name)


def get_output():
    """Get output."""
    print(f'Get output start, memory: {psutil.virtual_memory().percent}')
    with open(setting['NAME_LIST_FILE'], 'rb') as f_name:
        [probe_name_list, index_name_list] = pickle.load(f_name)
    sims = None
    for backbone, weight in zip(setting['MODEL_LIST'], setting['MODEL_WEIGHT']):
        pkl_name = os.path.join(setting['SIMS_DIR'], f'{backbone}_sims.pkl')
        with open(pkl_name, 'rb') as f_sims:
            backbone_sims = pickle.load(f_sims)
            print(f"backbone: {backbone}, weight: {weight}")
            if sims is None:
                sims = weight * backbone_sims
            else:
                sims += weight * backbone_sims
    print(f'Sim Fusion Done, memory: {psutil.virtual_memory().percent}')

    if setting['CATEGORY_RERANK'] == 'after_merge':
        sims = torch.tensor(sims).cuda().half()
        for idx, backbone in enumerate(setting['MODEL_LIST']):
            print('Computing category rerank after merge')
            probe_name_list, probe_feats = load_feat('probe', backbone)
            index_name_list, index_feats = load_feat('index', backbone)
            print(len(probe_name_list), len(index_name_list))
            ref_info = load_meta(backbone)
            ref_info['probe_name_list'] = probe_name_list
            ref_info['index_name_list'] = index_name_list
            probe_feats = torch.tensor(probe_feats).cuda().half()
            index_feats = torch.tensor(index_feats).cuda().half()
            index_tags, index_locs = get_index_tags(index_feats, ref_info, batch_size=128)
            probe_tags, probe_tag_scores, probe_locs, probe_loc_scores = get_probe_tags_topk(probe_feats, ref_info)
            sims = category_rerank_after_merge(sims, probe_tags, probe_locs,
                                               probe_tag_scores, probe_loc_scores,
                                               index_tags, index_locs,
                                               sim_thr=setting['CATEGORY_THR'],
                                               alpha=setting['alpha'],
                                               beta=setting['beta'])
        print(f'Tag Rerank for each model done!, memory: {psutil.virtual_memory().percent}')
        sims = sims.cpu().numpy()

    print(f'Get sims Memory: {psutil.virtual_memory().percent}')
    write_csv(probe_name_list, index_name_list, sims)


def main():
    """Main."""

    print(f'Init Memory usage: {psutil.virtual_memory().percent}')
    image_list, query_count = load_image_list()
    print(f'load image Memory usage: {psutil.virtual_memory().percent}')
    if image_list is None and query_count is None:
        print('Dummy submission!')
        shutil.copyfile(os.path.join(setting['IMAGE_DIR'],
                                     'sample_submission.csv'),
                        'submission.csv')
        return
    if setting['DEBUG_FLAG']:
        debug_reid_inference(image_list)
    if setting['CATEGORY_RERANK'] != 'off':
        # meta info shared by all models
        prepare_meta()
    for backbone in setting['MODEL_LIST']:
        if setting['REID_EXTRACT_FLAG']:
            reid = ReID_Inference(backbone)
            print(f'Load model, memory: {psutil.virtual_memory().percent}')
            start_time = time.time()
            feature_dic = reid.extract(image_list)
            print('%.4f s for %s' % ((time.time() - start_time), backbone))
            print(f'Extract feature Memory: {psutil.virtual_memory().percent}')
            save_feature(feature_dic, backbone)
            print(f'Save feature Memory: {psutil.virtual_memory().percent}')
        if setting['REF_SET_EXTRACT']:
            # These should be offline calculated.
            print('Extract refset feature')
            reid = ReID_Inference(backbone)
            print(f'Load model, memory: {psutil.virtual_memory().percent}')
            extract_refset(reid, backbone)
            print(f'Extract refset Memory: {psutil.virtual_memory().percent}')
        compute_sim(backbone)
    print(f'Compute sim Memory: {psutil.virtual_memory().percent}')
    get_output()


if __name__ == '__main__':
    main()

# ########################## Cell 8 Other setting  #########################


def gpu_argsort(temp):
    """Use torch for faster argsort."""

    temp = torch.from_numpy(temp).to('cuda').half()
    rank = torch.argsort(temp, dim=1).cpu().numpy()
    return rank

def mergesetfeat4(X):
    """Run FAC for one iteration."""

    torch.cuda.empty_cache()  # empty GPU memory
    beta1 = 0.2
    k1 = 5

    print(f'Memory usage: {psutil.virtual_memory().percent}')
    X = torch.tensor(X).cuda().half()
    all_num = len(X)
    S = torch.zeros((all_num, all_num)).half().cuda()
    for i in tqdm(range(0, X.shape[0])):
        sample_sim = torch.matmul(X[i][None, :], X.T).flatten()
        sample_rank = torch.argsort(sample_sim)
        S[i, sample_rank[:k1]] = torch.exp(sample_sim[sample_rank[:k1]]/beta1)
        S[i, i] = torch.exp(sample_sim[i]/beta1)    # this is duplicated???
    D_row = torch.sqrt(1. / torch.sum(S, axis=1))
    D_col = torch.sqrt(1. / torch.sum(S, axis=0))
    L = torch.outer(D_row, D_col) * S
    X = torch.matmul(L, X)
    X = X / torch.norm(X, 2, 1, keepdim=True)
    return X


def gcn_rerank_memory():
    """Kr rerank memory."""

    probe_name_list, probe_feats = load_feat('test')
    index_name_list, index_feats = load_feat('index')
    print(len(probe_name_list), len(index_name_list))
    data = np.vstack((probe_feats, index_feats))
    data = data.astype('float16')
    print(f'Memory usage: {psutil.virtual_memory().percent}')
    for _ in range(2):
        data = mergesetfeat4(data)
    prb_n = len(probe_name_list)
    probe_feats = data[:prb_n, :]
    index_feats = data[prb_n:, :]
    sims = torch.matmul(probe_feats, index_feats.T)
    sims = sims.cpu().numpy()
    write_csv(probe_name_list, index_name_list, sims)

# kr_rerank_memory()
# gcn_rerank_memory()
