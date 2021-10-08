"""Eval CBIR.
Author: gongyou.zyq
Date: 2020.11.25
"""

import os
import pickle
import shutil
import time

import cv2
import numpy as np


class CBIREvaluator():
    """CBIR Evaluator."""

    def __init__(self):
        self.query_instance_dic = pickle.load(open('GLDv2_search_label_competition_2021.pkl', 'rb'))
        self.selected_test_id = list(self.query_instance_dic.keys())
        self.result_dic = self.init_result_dic()
        self.search_dir = '../input/landmark-retrieval-2021/index/'
        self.query_dir = '../input/landmark-retrieval-2021/test/'
        self.VERBOSE = False
        self.MAP_METRIC = 'retrieval'
        self.VIS_FLAG = False
        self.VIS_TOPK = 20
        self.RANK_GLOBAL_TOPK = [1, 5, 10, 20, 100]

    @staticmethod
    def init_result_dic():
        """Set empty dic to cache results."""

        result_dic = {'gt_num_list': [], 'best_recall_list': [],
                      'upper_bound_list': [], 'best_thr_list': [],
                      'proposal_recall_list': [], 'pred_num_list': [],
                      'cmc_list': [], 'ap_list': [],
                      'prec_list': [], 'rec_list': [],
                      'out_prec_list': [], 'out_rec_list': []}
        return result_dic

    def output_final_result(self):
        """Output final result."""

        mean_largest_recall = np.mean(self.result_dic['best_recall_list'])
        mean_bound = np.mean(self.result_dic['upper_bound_list'])
        mean_thr = np.mean(self.result_dic['best_thr_list'])
        mean_gt_num = np.mean(self.result_dic['gt_num_list'])
        mean_ap = np.mean(self.result_dic['ap_list'])
        mean_proposal_recall = np.mean(self.result_dic['proposal_recall_list'],
                                       axis=0)
        mean_pred_num = np.mean(self.result_dic['pred_num_list'])
        # np.save('./tests/localizer/average_recall_%s' % \
        #         self._cfg.EVAL.SIM_MODE, mean_topk_recall)
        mean_cmc = np.mean(self.result_dic['cmc_list'], axis=0)
        mean_prec = np.mean(self.result_dic['out_prec_list'], axis=0)
        mean_rec = np.mean(self.result_dic['out_rec_list'], axis=0)
        mean_cmc = np.round(mean_cmc, 4)
        mean_prec = np.round(mean_prec, 4)
        mean_rec = np.round(mean_rec, 4)
        sim_mode = self.MAP_METRIC
        cmc_list = self.result_dic['cmc_list']
        print(f'----------Final Results for sim_mode: {sim_mode}------------')
        print(f'Total valid query num: {len(cmc_list)}')
        print('detection metric: ')
        print(f'average_gt_num: {mean_gt_num:.1f}, '
              f'average pred num: {mean_pred_num:.1f} '
              f'largest recall: {mean_largest_recall:.4f}, '
              f' average upper bound: {mean_bound:.1f}, '
              f'mean_thr: {mean_thr:.4f}')
        print(f'ranking metric for global {self.RANK_GLOBAL_TOPK}: ')
        print(f'CMC: {mean_cmc}, mAP: {mean_ap:.4f}')
        print(f'mean precision: {mean_prec}, mean recall: {mean_rec}')

    def log_info(self, info_str):
        """Log verbose info."""

        if self.VERBOSE:
            print(info_str)

    # pylint:disable=too-many-locals
    def eval_data(self, all_reid_info):
        """Eval data."""

        start_time = time.time()
        for query_instance_id in self.selected_test_id:
            self.log_info('----------eval query_instance_id: '
                          f'{query_instance_id}----------')
            if len(self.query_instance_dic[query_instance_id]) == 0:
                self.log_info('invalid query, skip eval this query')
                continue

            gt_info = self.load_gt_info(query_instance_id)
            gt_num = gt_info['gt_num']
            self.result_dic['gt_num_list'].append(gt_num)
            if gt_num == 0:
                self.log_info('gt_num=0, skip eval this query')
                continue
            pred_info = self.postprocess_pred_info(query_instance_id,
                                                   all_reid_info)
            res = self.get_matching_flag(gt_info, pred_info)
            [tp_flag, fp_flag, thrs, gt_matched_flag, valid_flag] = res
            rec, prec = self.get_pr(tp_flag, fp_flag, gt_num)
            if len(rec) == 0:
                print('empty pred, put all zeros')
                rec = np.array([0.0])
                prec, thrs, tp_flag = rec.copy(), rec.copy(), rec.copy()
            pad_lenth = 100
            if len(rec) < pad_lenth:
                # print('pad data')
                rec = np.pad(rec, (0, pad_lenth-len(rec)), 'edge')
                prec = np.pad(prec, (0, pad_lenth-len(prec)))
                thrs = np.pad(thrs, (0, pad_lenth-len(thrs)))
                tp_flag = np.pad(tp_flag, (0, pad_lenth-len(tp_flag)))
            unmatched_data_list = self.get_unmatched(query_instance_id,
                                                     gt_matched_flag)
            self.get_det_eval(rec, prec, thrs)
            self.get_rank_eval(rec, prec, tp_flag, gt_num)
            if self.VERBOSE:
                self.output_current_result(query_instance_id, tp_flag,
                                           valid_flag)

            if self.VIS_FLAG:
                trimmed_pred = [tp_flag, pred_info, valid_flag]
                self.vis_retrieval(query_instance_id, unmatched_data_list,
                                   trimmed_pred)
                print(f'{time.time() - start_time:.4f} seconds to eval all data')

    def output_current_result(self, query_instance_id, tp_flag, valid_flag):
        """Output current result."""

        matched_tp_index = np.argwhere(tp_flag > 0).flatten()
        print(f'matched tp index: {matched_tp_index}')
        sim_mode = self.MAP_METRIC
        best_recall = round(self.result_dic['best_recall_list'][-1], 4)
        upper_bound = round(self.result_dic['upper_bound_list'][-1], 4)
        best_thr = round(self.result_dic['best_thr_list'][-1], 4)
        gt_num = self.result_dic['gt_num_list'][-1]
        proposal_recall = self.result_dic['proposal_recall_list'][-1]
        cmc = self.result_dic['cmc_list'][-1]
        average_precision = self.result_dic['ap_list'][-1]
        out_prec = self.result_dic['out_prec_list'][-1]
        out_rec = self.result_dic['out_rec_list'][-1]
        print(f'sim_mode: {sim_mode}, data_shape: {valid_flag.shape}')
        print(f'best recall: {best_recall}, upper bound: {upper_bound}, '
              f'thr: {best_thr}, gt_num: {gt_num}, '
              f'proposal recall: {proposal_recall:.4f}')
        print(f'CMC: {cmc}, AP: {average_precision}')
        print(f'precision: {out_prec}, recall: {out_rec}')

    def load_gt_info(self, query_instance_id):
        """Load gt."""

        query_bbox_dic = self.query_instance_dic[query_instance_id]
        gt_bbox_dic = {}
        gt_matched_flag = {}
        gt_num = 0
        # query image should always be ignored whatever separate camera or not
        ignore_list = [query_bbox_dic['image_name']]

        gt_data_list = query_bbox_dic['pos_gallery_list']
        separate_cam = False
        for gt_data in gt_data_list:
            device_id = gt_data['device_id']
            image_name = gt_data['image_name']
            gt_bbox_dic[image_name] = gt_data['bbox']

            if gt_data['ignore']:
                ignore_list.append(image_name)
            if separate_cam and device_id != query_bbox_dic['device_id'] \
                    and not gt_data['ignore']:
                gt_num += 1
                gt_matched_flag[image_name] = 0
            if not separate_cam and not gt_data['ignore'] and\
                    image_name != query_bbox_dic['image_name']:
                gt_num += 1
                gt_matched_flag[image_name] = 0
            if image_name == query_bbox_dic['image_name']:
                ignore_list.append(image_name)
            if separate_cam and device_id == query_bbox_dic['device_id']:
                ignore_list.append(image_name)
        gt_info = {'gt_num': gt_num, 'gt_bbox_dic': gt_bbox_dic,
                   'gt_matched_flag': gt_matched_flag,
                   'ignore_list': ignore_list}
        return gt_info


    def load_local_proposal(self, loc_gallery_bbox_dic):
        """Keep topk per large image for eval localizer."""

        merged_bboxes = []
        merged_sims = []
        unique_image_ids = []
        repeat_times = []
        keep_num = 1
        for image_name, loc_pred_for_large in loc_gallery_bbox_dic.items():
            loc_pred_for_large = loc_gallery_bbox_dic[image_name]
            if len(loc_pred_for_large['sim']) == 0:
                continue
            indexes = np.argsort(-loc_pred_for_large['sim'])[:keep_num]
            merged_bboxes.append(loc_pred_for_large['bbox'][indexes])
            merged_sims.append(loc_pred_for_large['sim'][indexes])
            repeat_times.append(len(indexes))
            unique_image_ids.append(image_name)
        merged_bboxes = np.concatenate(merged_bboxes)
        merged_sims = np.concatenate(merged_sims)
        image_ids = np.repeat(unique_image_ids, repeat_times)
        return {'sim': merged_sims,
                'bbox': merged_bboxes,
                'image_name': image_ids}

    def postprocess_pred_info(self, query_instance_id, all_reid_info):
        """Postprocess pred info (How to modify proposal)."""

        pred_dic = all_reid_info[query_instance_id]
        pred_dic = self.load_local_proposal(pred_dic)
        pred_info = self.re_sort(pred_dic)
        return pred_info

    def re_sort(self, pred_dic):
        """Resort data."""

        # Ref: https://zhuanlan.zhihu.com/p/37910324
        pred_sim = pred_dic['sim']
        pred_bboxes = np.array(pred_dic['bbox'])
        image_ids = np.array(pred_dic['image_name'])

        sorted_ind = np.argsort(-pred_sim)
        sorted_sim = pred_sim[sorted_ind]
        pred_bboxes = pred_bboxes[sorted_ind, :]
        # image_ids = [image_ids[x] for x in sorted_ind]
        image_ids = image_ids[sorted_ind]
        pred_info = {'pred_bboxes': pred_bboxes, 'image_ids': image_ids,
                     'sorted_sim': sorted_sim}
        self.result_dic['pred_num_list'].append(len(pred_bboxes))
        return pred_info


    @staticmethod
    def get_pr(tp_flag, fp_flag, gt_num):
        """Get pr."""

        fp_flag = np.cumsum(fp_flag)
        tp_flag = np.cumsum(tp_flag)
        # if len(tp_flag) == 0:
        #     return 0.0, 0.0
        rec = tp_flag / float(gt_num)
        # avoid divide by zero in case the first detection matches
        # a difficult ground truth
        prec = tp_flag / np.maximum(tp_flag+fp_flag, np.finfo(np.float64).eps)
        # print(rec[:20])
        # print(prec[:20])
        return rec, prec

    def get_det_eval(self, rec, prec, thrs):
        """Get det eval"""

        rank_index = np.arange(1, len(rec)+1)
        best_recall = np.max(rec)
        upper_bound = np.min(rank_index[rec == best_recall])
        best_thr = thrs[upper_bound-1]
        proposal_recall = rec[-1]

        out_rec, out_prec = self.refine_pr(rec, prec)
        self.result_dic['best_recall_list'].append(best_recall)
        self.result_dic['upper_bound_list'].append(upper_bound)
        # Clip data for speed purpose
        self.result_dic['best_thr_list'].append(best_thr)
        self.result_dic['proposal_recall_list'].append(proposal_recall)
        self.result_dic['out_rec_list'].append(out_rec)
        self.result_dic['out_prec_list'].append(out_prec)
        self.result_dic['rec_list'].append(rec)
        self.result_dic['prec_list'].append(prec)

    def get_rank_eval(self, rec, prec, tp_flag, gt_num):
        """Get rank eval"""

        # NOTE: will clip ranklist by RANK_GLOBAL_TOPK
        if self.MAP_METRIC == 'delg':
            average_precision = self.delg_ap(tp_flag, gt_num)
        elif self.MAP_METRIC == 'voc':
            average_precision = self.voc_ap(rec, prec)
        elif self.MAP_METRIC == 'retrieval':
            average_precision = self.retrieval_ap(tp_flag, gt_num)
        else:
            print('Unknown ranking evaluation metric')
            return
        global_top_k = np.array(self.RANK_GLOBAL_TOPK) - 1
        max_eval_rank = self.RANK_GLOBAL_TOPK[-1]
        tp_flag = np.cumsum(tp_flag)
        if len(tp_flag) >= max_eval_rank:
            cmc = (tp_flag[:max_eval_rank] > 0).astype('int')
            cmc = cmc[global_top_k].astype('float32')
            self.result_dic['cmc_list'].append(cmc)
            self.result_dic['ap_list'].append(average_precision)
        else:
            print('too few predictions for ranking evaluation')

    def refine_pr(self, rec, prec):
        """Refine pr."""

        global_top_k = np.array(self.RANK_GLOBAL_TOPK) - 1
        out_rec = rec[global_top_k].astype('float32')
        out_prec = prec[global_top_k].astype('float32')
        # when compute for mAP, we use real precision and recall.
        # When compute for precision@K and recall@K, we foloow delg.
        # If `desired_pr_rank` is larger than last positive's rank,only compute
        # precision with respect to last positive's position.

        # pylint: disable=line-too-long
        # See https://github.com/tensorflow/models/blob/master/research/delf/delf/python/detect_to_retrieve/dataset.py for ComputePRAtRanks # noqa
        finish_indexes = np.argwhere(rec == 1.0).flatten()
        if len(finish_indexes) > 0:
            first_finish_index = finish_indexes.min()
            if first_finish_index <= global_top_k[1]:
                out_prec[1] = prec[first_finish_index]
                out_prec[2] = prec[first_finish_index]
                out_prec[3] = prec[first_finish_index]
            if global_top_k[1] < first_finish_index <= global_top_k[2]:
                out_prec[2] = prec[first_finish_index]
                out_prec[3] = prec[first_finish_index]
            if global_top_k[2] < first_finish_index <= global_top_k[3]:
                out_prec[3] = prec[first_finish_index]
        return out_rec, out_prec

    @staticmethod
    def voc_ap(rec, prec, use_07_metric=False):
        """Compute VOC AP given precision and recall. If use_07_metric is true,
        uses the VOC 07 11-point method (default:False).
        """

        # pylint: disable=invalid-name
        if use_07_metric:
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    @staticmethod
    def delg_ap(tp_flag, gt_num):
        """DELG official code for mAP, the convention for the Revisited
        Oxford/Paris datasets.
        """

        positive_ranks = np.where(tp_flag == 1)[0]
        average_precision = 0.0

        num_expected_positives = gt_num
        if not num_expected_positives:
            return average_precision

        recall_step = 1.0 / num_expected_positives
        for i, rank in enumerate(positive_ranks):
            if not rank:
                left_precision = 1.0
            else:
                left_precision = i / rank

            right_precision = (i + 1) / (rank + 1)
            average_precision += (left_precision + right_precision) * recall_step / 2

        return average_precision

    @staticmethod
    def retrieval_ap(tp_flag, gt_num):
        """Retrieval mAP, widely used in person reid.
        """

        positive_ranks = np.where(tp_flag == 1)[0]
        average_precision = 0.0

        num_expected_positives = gt_num
        if not num_expected_positives:
            return average_precision

        recall_step = 1.0 / num_expected_positives
        for i, rank in enumerate(positive_ranks):
            right_precision = (i + 1) / (rank + 1)
            average_precision += right_precision * recall_step

        return average_precision

    def _vis_query(self, query_large_path, query_bbox, save_dir):
        """Vis query."""

        img = self.draw_bbox(query_large_path, query_bbox, (255, 0, 0))
        if query_bbox is not None:
            cv2.putText(img, 'query',
                        (int(query_bbox[0]), int(query_bbox[1])),
                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(save_dir, 'query.jpg'), img)

    @staticmethod
    def draw_bbox(image_path, bbox, color):
        """Draw bbox."""

        img = cv2.imread(image_path)
        if bbox is not None:
            [x1, y1, x2, y2] = bbox    # pylint:disable=invalid-name
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        return img

    # pylint: disable=too-many-locals
    def _vis_gallery(self, trimmed_pred, save_dir):
        """Vis gallery."""

        [tp_flag, pred_info, valid_flag] = trimmed_pred
        bbox_list = pred_info['pred_bboxes'][valid_flag]
        sim = pred_info['sorted_sim'][valid_flag]
        image_ids = pred_info['image_ids'][valid_flag]
        for index, image_name in enumerate(image_ids):
            image_name = os.path.basename(image_name)
            short_dir = f'{image_name[0]}/{image_name[1]}/{image_name[2]}'
            sim_str = '%.4f' % sim[index]
            gal_bbox = bbox_list[index].astype('int')
            prefix = image_name.replace('/', '_')[:-4]
            string = f'{gal_bbox[0]}_{gal_bbox[1]}_{gal_bbox[2]}_{gal_bbox[3]}'
            save_name = f'{sim_str}_{prefix}_bbox_{string}.jpg'
            old_path = os.path.join(self.search_dir,
                                    short_dir,
                                    image_name)
            if not tp_flag[index] and index < self.VIS_TOPK:
                img = self.draw_bbox(old_path, gal_bbox, (255, 0, 0))
            elif tp_flag[index]:
                img = self.draw_bbox(old_path, gal_bbox, (0, 255, 0))
                save_name = save_name.split('.jpg')[0] + '_matched.jpg'
            else:
                continue
            cv2.putText(img, sim_str, (int(gal_bbox[0]), int(gal_bbox[1])),
                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(save_dir, save_name), img)

    def _vis_unmatched(self, unmatched_data_list, save_dir):
        """Vis unmatched results beyond top-k."""

        for unmatched_data in unmatched_data_list:
            image_name = unmatched_data['image_name']
            image_name = os.path.basename(image_name)
            short_dir = f'{image_name[0]}/{image_name[1]}/{image_name[2]}'
            save_name = 'missed_' + image_name.replace('/', '_')
            image_path = os.path.join(self.search_dir,
                                      short_dir,
                                      image_name)
            img = self.draw_bbox(image_path, unmatched_data['bbox'],
                                 (0, 0, 255))
            cv2.imwrite(os.path.join(save_dir, save_name), img)

    def get_unmatched(self, query_instance_id, gt_matched_flag):
        """Get unmatched gt."""

        query_bbox_dic = self.query_instance_dic[query_instance_id]
        gt_data_list = query_bbox_dic['pos_gallery_list']
        unmatched_gt_name_list = []
        for image_name in gt_matched_flag:
            if gt_matched_flag[image_name] == 0:
                unmatched_gt_name_list.append(image_name)
        unmatched_data_list = []
        for gt_data in gt_data_list:
            image_name = gt_data['image_name']
            if image_name in unmatched_gt_name_list:
                unmatched_data_list.append(gt_data)
        return unmatched_data_list

    def vis_retrieval(self, query_instance_id, unmatched_data_list,
                      trimmed_pred):
        """Vis retrieval."""

        image_name = self.query_instance_dic[query_instance_id]['image_name']
        image_name = os.path.basename(image_name)
        short_dir = f'{image_name[0]}/{image_name[1]}/{image_name[2]}'
        query_large_path = os.path.join(self.query_dir, short_dir, image_name)
        save_dir = f'./tests/images/vis_pred/{query_instance_id}/'
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        query_bbox = None
        self._vis_query(query_large_path, query_bbox, save_dir)
        self._vis_gallery(trimmed_pred, save_dir)
        self._vis_unmatched(unmatched_data_list, save_dir)

    def get_matching_flag(self, gt_info, pred_info):
        """Get matching flag"""

        image_ids = pred_info['image_ids']
        sorted_sim = pred_info['sorted_sim']
        ignore_list = gt_info['ignore_list']
        gt_bbox_dic = gt_info['gt_bbox_dic']
        gt_matched_flag = gt_info['gt_matched_flag']

        gt_name_list = list(gt_bbox_dic.keys())
        valid_flag = np.zeros(len(image_ids))
        tp_flag = valid_flag.copy()
        fp_flag = valid_flag.copy()
        thrs = valid_flag.copy()
        for rank, image_name in enumerate(image_ids):
            if image_name not in ignore_list:
                valid_flag[rank] = 1
            thrs[rank] = sorted_sim[rank]
            if image_name in gt_name_list:
                tp_flag[rank] = 1.
                gt_matched_flag[image_name] = 1
            else:
                fp_flag[rank] = 1.
        tp_flag = tp_flag[valid_flag > 0]
        fp_flag = fp_flag[valid_flag > 0]
        thrs = thrs[valid_flag > 0]
        valid_flag = np.where(valid_flag>0)[0]
        return tp_flag, fp_flag, thrs, gt_matched_flag, valid_flag


def main():
    """Main method"""

    all_reid_info = pickle.load(open('submission.pkl', 'rb'))

    cbir_evaluator = CBIREvaluator()
    cbir_evaluator.eval_data(all_reid_info)
    cbir_evaluator.output_final_result()


if __name__ == "__main__":
    main()
