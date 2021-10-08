"""Landmark retrieval offline code for ILR2021."""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import pickle
import os
import time
import sys
sys.path.append('../input/util-code/local_matching')
# from local_matching import load_superpointglue_model, generate_superpoint_superglue, get_num_inliers, get_total_score
# from superpointglue_util import get_whole_cached_num_inliers, save_whole_cached_num_inliers


# pylint: disable=not-callable, invalid-name, line-too-long
# ########################## Cell 2 Configs ##################################

MODE = 'retrieval'
setting = {
        # ############# General params ############
        'IMAGE_DIR': '../input/landmark-retrieval-2021/',
        'RAW_IMAGE_DIR': '/home/gongyou.zyq/datasets/google_landmark/',
        'OUTPUT_TEMP_DIR': f'../temp/{MODE}/',    # Will lose after reset
        'OUTPUT_DIR': f'../working/{MODE}/',    # Will be saved after reset
        'MODEL_DIR': '../input/models/',
        'META_DIR': '../input/meta-data/',
        'PROBE_DIR': '../input/landmark-retrieval-2021/test/',
        'INDEX_DIR': '../input/landmark-retrieval-2021/index/',
        'FEAT_DIR': f'../temp/{MODE}/features/',
        'SIMS_DIR': f'../temp/{MODE}/sims/',
        'SAMPLE_TEST_NUM': 1129,
        # ############# ReID params ############
        'REID_EXTRACT_FLAG': True,
        'FP16': True,
        'DEBUG_FLAG': False,
        'MULTI_SCALE_FEAT': False,
        # ############# ReID model list ############
        # 'MODEL_LIST': ['R50', 'R101ibn', 'RXt101ibn', 'SER101ibn', 'ResNeSt101', 'ResNeSt269', 'EffNetB7'],
        'MODEL_LIST': ['R50'],
        # 'MODEL_LIST': ['SER101ibn', 'RXt101ibn', 'ResNeSt101', 'ResNeSt269'],
        'MODEL_WEIGHT': [1.0, 1.0, 1.0, 1.0],
         #'MODEL_WEIGHT': [1.0,],
        'IMAGE_SIZE': 256,    # 256, 384, 448, 512
        'BATCH_SIZE': 128,
        'MODEL_PARAMS': {'R50': {'MODEL_NAME': 'R50_256.pth', 'BACKBONE': 'resnet50'},
                         'R101ibn': {'MODEL_NAME': 'R101ibn_384_finetune_c2x.pth', 'BACKBONE': 'resnet101_ibn_a'},
                         'RXt101ibn': {'MODEL_NAME': 'RXt101ibn_512_all.pth', 'BACKBONE': 'resnext101_ibn_a'},
                         'SER101ibn': {'MODEL_NAME': 'SER101ibn_512_all.pth', 'BACKBONE': 'se_resnet101_ibn_a'},
                         'ResNeSt101': {'MODEL_NAME': 'ResNeSt101_512_all.pth', 'BACKBONE': 'resnest101'},
                         'ResNeSt269': {'MODEL_NAME': 'ResNeSt269_512_all.pth', 'BACKBONE': 'resnest269'},
                         'EffNetB7': {'MODEL_NAME': 'efficientnet-b7_20_512_3796.pth', 'BACKBONE': 'efficientnet-b7'},
                         },
        # ############# Rerank params ############
        'KR_FLAG': False,
        'K1': 10,
        'K2': 3,
        'INITIAL_RANK_FILE': f'../temp/{MODE}/initial_rank.npy',
        'NAME_LIST_FILE': f'../temp/{MODE}/name_list.pkl',
        'EUC_DIST_DIR': f'../temp/{MODE}/euc_dist/',
        'GRAPH_DIST_DIR': f'../temp/{MODE}/graph_dist/',
        'QE_DIST_DIR': f'../temp/{MODE}/qe_dist/',
        'JACCARD_DIR': f'../temp/{MODE}/jaccard/',
        'LAMBDA': 0.3,
        # ############# Category Rerank ############
        'CATEGORY_RERANK': 'off',    # after_merge, before_merge or off
        'VOTE_NUM': 3,    # Soft voting seems not work
        'REF_SET_EXTRACT': False,    # Just need to cache once
        'REF_ALL_LIST': '../input/meta-data-final/cache_all_list.pkl',
        'REF_SET_LIST': '../input/meta-data-final/cache_full_list.pkl',    # full, index_train, all
        'REF_SET_META': f'../temp/{MODE}/ref_meta.pkl',
        'REF_SET_FEAT': '../input/meta-data-all/ref_feats.pkl',
        'REF_LOC_MAP': '../input/meta-data-final/gbid2country.pkl',
        'CATEGORY_THR': -1.0,
        'alpha': 1.0,
        'beta': 0.1,
        # ############ LocalMatching Rerank ############
        'LOCAL_MATCHING': 'off',    # 'spg' or 'off'
        'SPG_MODEL_DIR': '../input/models/local_matching',
        'SPG_CACHE_DIR': f'../temp/{MODE}/local_matching_cache',
        'SPG_RERANK_NUM': 10,    # rerank length, larger is better
        'LOCAL_WEIGHT': 0.15,
        'MAX_INLIERS': 90,
        'SPG_DO_CACHE': True,    # wheather save inliers cache or not.
        }


# ########################## Cell 8 Get output file  #########################

def slice_jaccard(probe_feat, topk_index_feats):
    """Kr rerank for only top-k index feats."""

    query_num = 1
    gallery_num = len(topk_index_feats)
    all_num = query_num + gallery_num
    concat_feat = torch.cat([probe_feat, topk_index_feats])
    cos_sim = torch.matmul(concat_feat, concat_feat.T)    # (101, 101)
    original_dist = 1.0 - (cos_sim + 1.0)/2
    initial_rank = torch.argsort(original_dist, dim=1)
    initial_rank = initial_rank.cpu().numpy()
    original_dist = original_dist.cpu().numpy()
    # print(f'Memory usage: {psutil.virtual_memory().percent}')
    V = np.zeros((all_num, all_num))
    gallery_num = original_dist.shape[0]

    k1 = setting['K1']
    k2 = setting['K2']
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = weight/np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2-temp_min)

    # final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    slice_jaccard = jaccard_dist[:query_num,query_num:].flatten()
    return torch.tensor(slice_jaccard).cuda().half()


def kr_rerank_fast(probe_feats, index_feats):
    """Memory efficient rerank.

    probe_feats and index_feats are in gpu tensor.
    """

    # print('Starting KR re_ranking')
    topk = 30
    fast_lambda = 0.5
    cos_sims = torch.matmul(probe_feats, index_feats.T)
    original_dists = 1.0 - (cos_sims + 1.0)/2
    query_num = len(probe_feats)
    gallery_num = len(index_feats)
    final_dists = torch.zeros((query_num, gallery_num)).cuda().half()
    # for i, probe_feat in enumerate(tqdm(probe_feats)):
    for i, probe_feat in enumerate(probe_feats):
        q_sim = cos_sims[i]
        _, top_indices = torch.topk(q_sim, topk)
        q_neighbour_feats = index_feats[top_indices]
        q_neighbour_sims = torch.matmul(q_neighbour_feats, index_feats.T)    # (topk, 400w)
        _, neighbour_top_indices = torch.topk(q_neighbour_sims, topk)    # (topk, topk)
        neighbour_top_indices = neighbour_top_indices.flatten()
        top_indices = torch.cat([top_indices, neighbour_top_indices])
        top_indices = torch.unique(top_indices)
        # print(q_neighbour_sims.shape, neighbour_top_indices.shape, top_indices.shape)
        topk_index_feats = index_feats[top_indices]
        jaccard_dist = slice_jaccard(probe_feat[None, :], topk_index_feats)
        expand_jaccard = torch.ones((gallery_num,)).cuda().half()
        expand_jaccard[top_indices] = jaccard_dist
        final_dists[i] = original_dists[i] * fast_lambda
        final_dists[i] += expand_jaccard * (1-fast_lambda)
    return 1.0 - final_dists


def merge_tags(tags_list, scores, weight):
    if len(tags_list)==1 or scores is None:
        return tags_list[-1]

    tags = torch.tensor(tags_list)
    scores = torch.tensor(scores)
    if weight is not None:
        weight = torch.tensor(weight)
    print(tags.shape, scores.shape)
    #print(tags_list.shape)
    #tags = torch.stack(tags_list, dim=0)
    merged_tags = []
    count = 0
    for i in range(tags.shape[1]):
        preds = torch.unique(tags[:, i])
        if weight is not None:
            score = scores[:, i] * weight
        else:
            score = scores[:, i]
        if len(preds) == 1:
            merged_tags.append(preds[0])
        elif len(preds) == tags.shape[0]:
            merged_tags.append(-1)
        else:
            unique_score_list = []
            for item in preds:
                sum_score = torch.sum(score[tags[:, i] == item])
                unique_score_list.append(sum_score)
            unique_score_list = torch.tensor(unique_score_list)
            best_index = torch.argmax(unique_score_list)
            merged_tags.append(preds[best_index])
            #print(f"{tags[:, i]}->{preds[best_index]}")
            #print(f"{score}")
            count += 1
    print(f'{count} low constancy tags')
    return merged_tags

def get_probe_tags_index(probe_feats, index_feats, index_tags, mode='avg'):

    probe_tags = []
    probe_scores = []
    if mode == 'avg':
        tag_mean_feats = []
        index_tags_unique = torch.unique(index_tags)
        for index_tag in index_tags_unique:
            same_tags = torch.where(index_tags == index_tag)
            same_tag_feats = torch.mean(index_feats[same_tags], dim=0, keepdim=True)
            same_tag_feats = same_tag_feats / torch.norm(same_tag_feats, 2, 1)
            tag_mean_feats.append(same_tag_feats)
        tag_mean_feats = torch.cat(tag_mean_feats, dim=0)

        for probe_index, query_feat in enumerate(tqdm(probe_feats)):
            sim = torch.matmul(tag_mean_feats, query_feat[:, None]).flatten()
            _, indices = torch.topk(sim, 1)
            probe_tag = index_tags_unique[indices[0]]
            probe_tags.append(probe_tag)
            probe_scores.append(sim[indices[0]])
            print(sim[indices[0]])
    elif mode == 'single':
        for probe_index, query_feat in enumerate(tqdm(probe_feats)):
            sim = torch.matmul(index_feats, query_feat[:, None]).flatten()
            _, indices = torch.topk(sim, 1)
            probe_tag = index_tags[indices[0]]
            probe_tags.append(probe_tag)
            probe_scores.append(sim[indices[0]])

    return probe_tags, probe_scores

def get_probe_tags_avg(probe_feats, ref_info):
    globalid2refindex = ref_info['globalid2refindex']
    ref_feats = ref_info['ref_feats']
    category_mean_feats = []
    for globalid in sorted(globalid2refindex.keys()):
        refindexes = globalid2refindex[globalid]
        same_id_feats = ref_feats[refindexes]
        same_id_feats = torch.mean(same_id_feats, dim=0, keepdim=True)
        same_id_feats = same_id_feats / torch.norm(same_id_feats, 2, 1)
        category_mean_feats.append(same_id_feats)
    category_mean_feats = torch.cat(category_mean_feats, dim=0)
    probe_tags = []

    print(f'computing probe tags, total probes:{probe_feats.shape[0]}, refs: {ref_feats.shape[0]}')
    for probe_index, query_feat in enumerate(tqdm(probe_feats)):
        ref_sim = torch.matmul(category_mean_feats, query_feat[:, None]).flatten()
        _, ref_indices = torch.topk(ref_sim, setting['VOTE_NUM'])
        pred_global_id = ref_indices[0]
        probe_tags.append(pred_global_id)

    return probe_tags


def get_probe_tags_topk(probe_feats, ref_info):
    """Get topk probe tags in gpu tensor."""

    ref_gbid = ref_info['ref_gbid']
    ref_loc = ref_info['ref_loc']
    ref_feats = ref_info['ref_feats']
    probe_tags = []
    probe_tag_scores = []
    probe_locs = []
    probe_loc_scores = []
    print(f'computing probe tags, total probes:{probe_feats.shape[0]}, refs: {ref_feats.shape[0]}')
    for probe_index, query_feat in enumerate(tqdm(probe_feats)):
        # ref_sim = torch.matmul(ref_feats, query_feat[:, None]).flatten()
        ref_sim = kr_rerank_fast(query_feat[None, :], ref_feats).flatten()
        _, ref_indices = torch.topk(ref_sim, setting['VOTE_NUM'])

        id_list = ref_gbid[ref_indices]
        loc_list = ref_loc[ref_indices]
        score_list = ref_sim[ref_indices]
        unique_id_list = torch.unique(id_list)
        unique_loc_list = torch.unique(loc_list)
        id_score_list = []
        loc_score_list = []
        for item in unique_id_list:
            indexes = torch.where(id_list == item)[0]
            sum_score = torch.sum(score_list[indexes])
            id_score_list.append(sum_score)
        for item in unique_loc_list:
            indexes = torch.where(loc_list == item)[0]
            sum_score = torch.sum(score_list[indexes])
            loc_score_list.append(sum_score)

        id_score_list = torch.tensor(id_score_list)
        id_score_list = id_score_list / torch.sum(id_score_list)

        loc_score_list = torch.tensor(loc_score_list)
        loc_score_list = loc_score_list / torch.sum(loc_score_list)

        probe_tags.append(unique_id_list)
        probe_tag_scores.append(id_score_list)
        probe_locs.append(unique_loc_list)
        probe_loc_scores.append(loc_score_list)

    return probe_tags, probe_tag_scores, probe_locs, probe_loc_scores

def get_probe_tags(probe_feats, ref_info):
    ref_gbid = []
    refindex2globalid = ref_info['refindex2globalid']
    for refindex in refindex2globalid:
        ref_gbid.append(refindex2globalid[refindex])
    ref_gbid = torch.tensor(ref_gbid).cuda()

    ref_feats = ref_info['ref_feats']
    probe_tags = []
    probe_scores = []

    print(f'computing probe tags, total probes:{probe_feats.shape[0]}, refs: {ref_feats.shape[0]}')
    for probe_index, query_feat in enumerate(tqdm(probe_feats)):
        ref_sim = torch.matmul(ref_feats, query_feat[:, None]).flatten()
        # ref_sim = kr_rerank_fast(query_feat[None, :], ref_feats).flatten()
        _, ref_indices = torch.topk(ref_sim, setting['VOTE_NUM'])

        pred_id_list = []
        pred_score_list = []
        for ref_index in ref_indices:
            pred_score = ref_sim[ref_index]
            pred_global_id = ref_gbid[ref_index]
            pred_id_list.append(pred_global_id)
            pred_score_list.append(pred_score)
        pred_id_list = torch.tensor(pred_id_list)
        pred_score_list = torch.tensor(pred_score_list)

        if len(torch.unique(pred_id_list)) == 1:
            # This is often the case
            pred_global_id = pred_id_list[0]
            score = torch.sum(pred_score_list)
        else:
            unique_id_list = torch.unique(pred_id_list)
            unique_score_list = []
            for item in unique_id_list:
                indexes = torch.where(pred_id_list == item)[0]
                sum_score = torch.sum(pred_score_list[indexes])
                unique_score_list.append(sum_score)
            unique_score_list = torch.tensor(unique_score_list)
            best_index = torch.argmax(unique_score_list)
            pred_global_id = unique_id_list[best_index]
            score = unique_score_list[best_index]
        probe_tags.append(pred_global_id)
        probe_scores.append(score)
    return probe_tags, probe_scores


def get_index_tags(index_feats, ref_info, batch_size=128):

    ref_feats = ref_info['ref_feats']
    ref_gbid = ref_info['ref_gbid']
    ref_loc = ref_info['ref_loc']

    print(f'computing index tags, total index:{index_feats.shape[0]}, refs: {ref_feats.shape[0]}')
    num_batches = len(index_feats) / batch_size + 1
    num_batches = int(num_batches)
    index_gbid = []
    index_locs = []
    for batch_idx in tqdm(range(num_batches)):
        batch_data = index_feats[batch_idx*batch_size:(batch_idx+1)*batch_size]
        ref_sim = torch.matmul(batch_data, ref_feats.T)
        _, ref_indices = torch.topk(ref_sim, 1, dim=1)
        ref_indices = ref_indices.flatten()    # (batch_size, )
        pred_global_id = ref_gbid[ref_indices]    # (batch_size, )
        pred_loc = ref_loc[ref_indices]
        index_gbid.append(pred_global_id)
        index_locs.append(pred_loc)
    index_gbid = torch.cat(index_gbid)    # (num_index, )
    index_locs = torch.cat(index_locs)

    return index_gbid, index_locs

def rerank_tag_and_loc(sim, probe_tags, probe_locs, probe_tag_scores, probe_loc_scores, index_tags, index_locs, alpha=1.0, beta=1.0):


    for idx, (probe_tag, probe_tag_score) in enumerate(zip(probe_tags, probe_tag_scores)):
        good_tag_indexes = torch.where(index_tags == probe_tag)

        sim = recomputing_sim(sim, good_tag_indexes, probe_tag_score, alpha)

    for idx, (probe_loc, probe_loc_score) in enumerate(zip(probe_locs, probe_loc_scores)):
        good_loc_indexes = torch.where(index_locs == probe_loc)

        sim = recomputing_sim(sim, good_loc_indexes, probe_loc_score, beta)


    return sim

def recomputing_sim(sim, indexes, score, weight):

    sim[indexes] += weight * score

    return sim

def category_rerank_after_merge(sims, probe_tags, probe_locs, probe_tag_scores, probe_loc_scores, index_tags, index_locs, sim_thr=0.1, alpha=1.0, beta=0.1):
    """Category rerank."""

    print('Category Reranking after merge......')
    print(f'Category Thr is {sim_thr}')
    rerank_sims = torch.zeros_like(sims)
    print(f'rerank sims by {alpha}, {beta}')
    for probe_index, (probe_tag, probe_loc, probe_tag_score, probe_loc_score) in enumerate(tqdm(zip(probe_tags, probe_locs, probe_tag_scores, probe_loc_scores))):

        # print(probe_tag, probe_loc, probe_tag_score, probe_loc_score)

        raw_sim = sims[probe_index].flatten()
        rerank_sims[probe_index] = rerank_tag_and_loc(raw_sim,
                                                      probe_tag, probe_loc, probe_tag_score, probe_loc_score,
                                                      index_tags, index_locs,
                                                      alpha=alpha, beta=beta)
    return rerank_sims


def category_rerank_before_merge(probe_feats, index_feats, ref_info):
    """Category rerank."""

    print('Category Reranking before merge......')
    ref_feats = ref_info['ref_feats']
    print('ref, ', ref_feats.shape)
    rerank_sims = np.zeros((len(probe_feats), len(index_feats)),
                           dtype=np.float32)
    index_gbid = []
    ref_gbid = []
    refindex2globalid = ref_info['refindex2globalid']
    for refindex in refindex2globalid:
        ref_gbid.append(refindex2globalid[refindex])
    ref_gbid = torch.tensor(ref_gbid).cuda()
    print('Get label for each index image')
    batch_size = 128
    num_batches = len(index_feats) / batch_size + 1
    num_batches = int(num_batches)
    for batch_idx in tqdm(range(num_batches)):
        batch_data = index_feats[batch_idx*batch_size:(batch_idx+1)*batch_size]
        ref_sim = torch.matmul(batch_data, ref_feats.T)
        _, ref_indices = torch.topk(ref_sim, 1, dim=1)
        ref_indices = ref_indices.flatten()    # (batch_size, )
        pred_global_id = ref_gbid[ref_indices]    # (batch_size, )
        index_gbid.append(pred_global_id)
    index_gbid = torch.cat(index_gbid)    # (num_index, )

    for probe_index, query_feat in enumerate(tqdm(probe_feats)):
        ref_sim = torch.matmul(ref_feats, query_feat[:, None]).flatten()
        _, ref_indices = torch.topk(ref_sim, setting['VOTE_NUM'])

        pred_id_list = []
        pred_score_list = []
        for ref_index in ref_indices:
            pred_score = ref_sim[ref_index]
            pred_global_id = ref_gbid[ref_index]
            pred_id_list.append(pred_global_id)
            pred_score_list.append(pred_score)
        pred_id_list = torch.tensor(pred_id_list)
        pred_score_list = torch.tensor(pred_score_list)

        if len(torch.unique(pred_id_list)) == 1:
            # This is often the case
            pred_global_id = pred_id_list[0]
        else:
            unique_id_list = torch.unique(pred_id_list)
            unique_score_list = []
            for item in unique_id_list:
                indexes = torch.where(pred_id_list == item)[0]
                sum_score = torch.sum(pred_score_list[indexes])
                unique_score_list.append(sum_score)
            unique_score_list = torch.tensor(unique_score_list)
            best_index = torch.argmax(unique_score_list)
            pred_global_id = unique_id_list[best_index]
            # print(pred_id_list, pred_score_list, pred_global_id)

        raw_sim = torch.matmul(index_feats, query_feat[:, None]).flatten()
        raw_orders = torch.argsort(-raw_sim)
        raw_orders = raw_orders.cpu().numpy()
        good_indexes = torch.where(index_gbid == pred_global_id)[0]
        good_indexes = good_indexes.cpu().numpy()
        match_indexes = np.in1d(raw_orders, good_indexes)
        pos_list = list(raw_orders[match_indexes])
        neg_list = list(raw_orders[~match_indexes])
        #pos_list = list(good_indexes)
        #neg_list = list(np.arange(index_feats.shape[0])[~np.in1d(np.arange(index_feats.shape[0]), good_indexes)])
        merged_list = pos_list + neg_list
        dummpy_sim = np.arange(len(merged_list)) / float(len(merged_list))
        dummpy_sim = 1.0 - dummpy_sim
        rerank_sims[probe_index, merged_list] = dummpy_sim
    return rerank_sims


def category_expansion(probe_feats, index_feats, ref_info):
    """Category query expansion."""

    ref_feats = ref_info['ref_feats']
    rerank_sims = np.zeros((len(probe_feats), len(index_feats)),
                           dtype=np.float32)
    for probe_index, query_feat in enumerate(tqdm(probe_feats)):
        globalid2refindex = ref_info['globalid2refindex']
        refindex2globalid = ref_info['refindex2globalid']
        ref_sim = torch.matmul(ref_feats, query_feat[:, None]).flatten()
        _, ref_indices = torch.topk(ref_sim, setting['VOTE_NUM'])
        ref_indices = ref_indices.cpu().numpy()
        same_cat_indexes = []
        for ref_index in ref_indices:
            pred_global_id = refindex2globalid[ref_index]
            same_cat_indexes.append(globalid2refindex[pred_global_id])
        same_cat_indexes = np.concatenate(same_cat_indexes)
        same_cat_indexes = torch.tensor(same_cat_indexes)
        # print(same_cat_indexes)
        cat_feats = ref_feats[same_cat_indexes]
        # print(f'{len(cat_feats)} ref images with same cat, {cat_feats.shape}')
        cat2pred_sim = torch.matmul(cat_feats, index_feats.T)    # (C, index_num)
        cat2pred_sim, _ = torch.max(cat2pred_sim, dim=0)    # (index_num, )
        # print(cat2pred_sim.shape, cat2pred_sim.max())
        good_indexes = (cat2pred_sim > 0.6).nonzero()
        good_indexes = good_indexes.cpu().numpy().flatten()
        # back to index_name_list
        # good_names = index_name_list[good_indexes]
        # print(f'good names: {good_names}')

        # 2019 GLR retrieval rerank
        raw_sim = torch.matmul(index_feats, query_feat[:, None]).flatten()
        raw_orders = torch.argsort(-raw_sim)
        raw_orders = raw_orders.cpu().numpy()
        match_indexes = np.in1d(raw_orders, good_indexes)
        pos_list = list(raw_orders[match_indexes])
        neg_list = list(raw_orders[~match_indexes])

        merged_list = pos_list + neg_list
        # merged_list = list(raw_orders)
        dummpy_sim = np.arange(len(merged_list)) / float(len(merged_list))
        dummpy_sim = 1.0 - dummpy_sim
        rerank_sims[probe_index, merged_list] = dummpy_sim
        """
        # simply query expansion max sims.
        # rerank_sims[probe_index] = cat2pred_sim.cpu().numpy()

        # QE average query features
        # qe_feats = ref_feats[same_cat_indexes]
        # qe_feats = torch.mean(qe_feats, dim=0, keepdim=True)    # (1, 512)
        # qe_feats = qe_feats / torch.norm(qe_feats, 2, 1)
        # qe2index_sim = torch.matmul(qe_feats, index_feats.T)    # (1, index_num)
        # rerank_sims[probe_index] = qe2index_sim.flatten().cpu().numpy()
        """
    return rerank_sims


def rerank_local_matching(spg_model, num_inliers_dict, probe_name, probe_dir, index_name_list, index_dir, sims, local_weight, max_inliers, cache_dir, do_cache, ignore_global_score=False):
    if do_cache:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    probe_path = f'{probe_dir}/{probe_name[0]}/{probe_name[1]}/{probe_name[2]}/{probe_name}.jpg'
    scores = []
    probe_image_cache = {}
    keypoint_time = 0
    spp_time = 0
    spg_time = 0
    matching_time = 0
    for idx, index_name in enumerate(index_name_list):
        index_path = f'{index_dir}/{index_name[0]}/{index_name[1]}/{index_name[2]}/{index_name}.jpg'

        if (probe_name, index_name) not in num_inliers_dict:
            start = time.time()
            pred, spp_t, spg_t = generate_superpoint_superglue(probe_path, probe_name, index_path, index_name,
                                                 spg_model, cache_dir, False, probe_image_cache)

            spp_time += spp_t
            spg_time += spg_t
            end_keypoint = time.time()
            keypoint_time += (end_keypoint - start)

            num_inliers = get_num_inliers(pred)
            matching_time += (time.time() - end_keypoint)

            num_inliers_dict[(probe_name, index_name)] = num_inliers
        else:
            num_inliers = num_inliers_dict.get((probe_name, index_name))
        if ignore_global_score:
            total_score = get_total_score(num_inliers, 0.)
        else:
            total_score = get_total_score(num_inliers, sims[idx], weight=local_weight, max_inlier_score=max_inliers)

        if False and idx % 9 == 0 and idx != 0:
            print(f"time of extract keypoints: {keypoint_time/idx}")
            print(f"time of extract SPP keypoints: {spp_time/idx}")
            print(f"time of matching SPG: {spg_time/idx}")
            print(f"time of matching: {matching_time/idx}")
        scores.append(total_score)

    #if do_cache:
    #    save_whole_cached_num_inliers(cache_dir, num_inliers_dict)
    scores = np.asarray(scores)
    rerank_sort = np.argsort(scores)[::-1]
    return index_name_list[rerank_sort]



def write_csv(probe_name_list, index_name_list, sims):
    """Write csv files for submission."""

    if setting['LOCAL_MATCHING'] == 'spg':
        spg_model = load_superpointglue_model(setting['SPG_MODEL_DIR'])
        num_inliers_dict = get_whole_cached_num_inliers(setting['SPG_CACHE_DIR'])
        #num_inliers_dict = {}
        rerank_num = setting['SPG_RERANK_NUM']
    index_name_list = np.array(index_name_list)
    id_list = []
    res_list = []
    print('Start output csv files')
    for probe_index, probe_name in enumerate(tqdm(probe_name_list)):
        id_list.append(probe_name)
        sim = sims[probe_index]
        orders = np.argsort(-sim)
        if setting['LOCAL_MATCHING'] == 'spg':
            sorted_name_list_topk = rerank_local_matching(spg_model, num_inliers_dict,
                                                        probe_name, setting['PROBE_DIR'],
                                                        index_name_list[orders[:rerank_num]], setting['INDEX_DIR'],
                                                        sim[orders[:rerank_num]], setting['LOCAL_WEIGHT'], setting['MAX_INLIERS'],
                                                        cache_dir=setting['SPG_CACHE_DIR'], do_cache=setting['SPG_DO_CACHE'])
            sorted_name_list = sorted_name_list_topk.tolist() + index_name_list[orders[rerank_num:100]].tolist()
        else:
            sorted_name_list = index_name_list[orders[:100]]
        res_str = ''
        for item in sorted_name_list:
            res_str += item + ' '
        res_str = res_str[:-1]
        res_list.append(res_str)
    # pylint: disable=invalid-name
    if setting['LOCAL_MATCHING']=='spg' and setting['SPG_DO_CACHE']:
        save_whole_cached_num_inliers(setting['SPG_CACHE_DIR'], num_inliers_dict)
    df = pd.DataFrame({'id': id_list, 'images': res_list})
    df.to_csv('submission.csv', index=False)
    print('Finish output csv files')
