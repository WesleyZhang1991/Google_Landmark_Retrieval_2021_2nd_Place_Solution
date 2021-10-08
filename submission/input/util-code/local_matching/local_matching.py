import pydegensac
import copy
import numpy as np
import time
from superpointglue_util import read_image as spg_read_image
from superpointglue_util import load_cached_keypoints, load_cached_matches, save_cached_keypoints, save_cached_matches

NUM_PUBLIC_TRAIN_IMAGES = 1580470

# RANSAC parameters:
MAX_INLIER_SCORE = 70
MAX_REPROJECTION_ERROR = 4.0
MAX_RANSAC_ITERATIONS = 1000
HOMOGRAPHY_CONFIDENCE = 0.99

def compute_num_inliers(test_keypoints, test_descriptors, train_keypoints,
                        train_descriptors, do_kdtree=True):
  """Returns the number of RANSAC inliers."""

  if do_kdtree:
    test_match_kp, train_match_kp = compute_putative_matching_keypoints(
        test_keypoints, test_descriptors, train_keypoints, train_descriptors)
  else:
    test_match_kp, train_match_kp = test_keypoints, train_keypoints
  if test_match_kp.shape[0] <= 4:  # Min keypoints supported by `pydegensac.findHomography()`
    return 0

  try:
    _, mask = pydegensac.findHomography(test_match_kp, train_match_kp,
                                        MAX_REPROJECTION_ERROR,
                                        HOMOGRAPHY_CONFIDENCE,
                                        MAX_RANSAC_ITERATIONS)
  except np.linalg.LinAlgError:  # When det(H)=0, can't invert matrix.
    return 0
  return int(copy.deepcopy(mask).astype(np.float32).sum())

def get_total_score(num_inliers, global_score, weight=1.0, max_inlier_score=None):
    if max_inlier_score is None:
        max_inlier_score = MAX_INLIER_SCORE
    local_score = min(num_inliers, max_inlier_score) / max_inlier_score
    return local_score*weight + global_score


def load_superpointglue_model(model_dir):

    from superpointglue.matching import Matching

    config = {
    'superpoint': {
      'nms_radius': 4,
      'keypoint_threshold': 0.005,
      'max_keypoints': 1024,
      'model_dir': model_dir,
    },
    'superglue': {
      'weights': 'outdoor',  # indoor, outdoor
      'sinkhorn_iterations': 20,
      'match_threshold': 0.2,
      'model_dir': model_dir,
    }
    }
    superpointglue = Matching(config).eval().cuda()
    return superpointglue

def generate_superpoint_superglue(test_image_path, test_image_id, train_image_path, train_image_id, superpointglue_net, cache_dir, do_cache=True, test_image_cache=None):

    if test_image_id not in test_image_cache:
        test_image, test_inp, test_scales = spg_read_image(test_image_path, resize=[800], rotation=0, resize_float=False)
        test_keypoints, test_scores, test_descriptors = None, None, None
    else:
        test_image, test_inp, test_scales, test_keypoints, test_scores, test_descriptors = test_image_cache[test_image_id]
    train_image, train_inp, train_scales = spg_read_image(train_image_path, resize=[800], rotation=0, resize_float=False)


    data_inp = {'image0': test_inp, 'image1': train_inp}
    if test_keypoints is not None:
        data_inp = {**data_inp, **{'keypoints0': test_keypoints, 'scores0': test_scores, 'descriptors0': test_descriptors}}
    pred, extract_time, matching_time = superpointglue_net(data_inp)

    test_keypoints, test_scores, test_descriptors = pred['keypoints0'], pred['scores0'], pred['descriptors0']
    train_keypoints, train_scores, train_descriptors = pred['keypoints1'], pred['scores1'], pred['descriptors1']
    test_train_matches0, test_train_matches1 = pred['matches0'], pred['matches1']
    test_train_matching_scores0, test_train_matching_scores1 = pred['matching_scores0'], pred['matching_scores1']
    if do_cache:
        save_cached_keypoints(cache_dir, test_image_id, test_keypoints, test_scores, test_descriptors, test_scales)
        save_cached_keypoints(cache_dir, train_image_id, train_keypoints, train_scores, train_descriptors, train_scales)
        save_cached_matches(cache_dir, test_image_id, train_image_id, test_train_matches0,
                        test_train_matches1, test_train_matching_scores0, test_train_matching_scores1)

    test_image_cache[test_image_id] = (test_image, test_inp, test_scales, test_keypoints, test_scores, test_descriptors)
    pred['scales0'] = test_scales
    pred['scales1'] = train_scales
    return pred, extract_time, matching_time

def get_num_inliers(pred):

    query_scales = pred['scales0']
    query_keypoints = copy.deepcopy(pred['keypoints0'])[0].cpu().numpy()
    support_scales = pred['scales1']
    support_keypoints = copy.deepcopy(pred['keypoints1'])[0].cpu().numpy()

    matches = pred['matches0'].cpu().numpy()[0]

    query_keypoints = query_keypoints * np.array([list(query_scales)])
    query_keypoints = query_keypoints[:, ::-1]

    support_keypoints = support_keypoints * np.array([list(support_scales)])
    support_keypoints = support_keypoints[:, ::-1]

    valid = matches > -1
    query_keypoints = query_keypoints[valid]
    support_keypoints = support_keypoints[matches[valid]]

    num_inliers = compute_num_inliers(query_keypoints, None, support_keypoints, None, do_kdtree=False)
#    print('num_inliers:', num_inliers)

    return num_inliers

def do_local_matching(model_dir):

    pass


if __name__ == '__main__':

    superpoint_model_dir = '/home/xianzhe.xxz/projects/instance_level_recognition/SuperGluePretrainedNetwork/models'
    superpoint_model = load_superpointglue_model(superpoint_model_dir)
    img1 = '00016575233bc956.jpg'
    img2 = '0002c06b2440a5f9.jpg'
    pred = generate_superpoint_superglue(img1, img2, superpoint_model)
    rescore_and_rerank_by_num_inliers(img1, img2, superpoint_model)
    print(pred.keys())
    print(pred['matches0'].shape)
    print(pred['matches1'].shape)

