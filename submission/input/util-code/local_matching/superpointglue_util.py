import cv2
import numpy as np
import torch
import pickle
from os.path import exists as ope

def get_whole_cached_num_inliers(cache_dir):

    whole_ransac_fname = f'{cache_dir}/whole_ransac_inliers.pkl'
    if ope(whole_ransac_fname):
        print(f'loading inliers from {whole_ransac_fname}')
        with open(whole_ransac_fname, 'rb') as dbfile:
            data = pickle.load(dbfile)
    else:
        data = dict()
    return data

def save_whole_cached_num_inliers(cache_dir, data):
  whole_ransac_fname = f'{cache_dir}/whole_ransac_inliers.pkl'
  with open(whole_ransac_fname, 'wb') as dbfile:
    pickle.dump(data, dbfile)


def load_cached_keypoints(keypoint_cache_dir, img_id):
    keypoint_fname = f'{keypoint_cache_dir}/keypoint_{img_id}.pkl'
    if ope(keypoint_fname):
        with open(keypoint_fname, 'rb') as dbfile:
            data = pickle.load(dbfile)
        return data
    else:
        return None

def save_cached_keypoints(keypoint_cache_dir, img_id, keypoints, scores, descriptors, scales):
    keypoint_fname = f'{keypoint_cache_dir}/keypoint_{img_id}.pkl'
    if not ope(keypoint_fname):
        data = {
        'keypoints': keypoints[0].cpu().numpy(),
        'scores': scores[0].data.cpu().numpy(),
        'descriptors': descriptors[0].data.cpu().numpy(),
        'scales': scales,
        }
        with open(keypoint_fname, 'wb') as dbfile:
            pickle.dump(data, dbfile)

def load_cached_matches(keypoint_cache_dir, query_image_id, index_image_id):
    match_fname = f'{keypoint_cache_dir}/match_query_{query_image_id}_index_{index_image_id}.pkl'
    if ope(match_fname):
        try:
            with open(match_fname, 'rb') as dbfile:
                data = pickle.load(dbfile)
        except:
            data = None
        return data
    else:
        return None

def save_cached_matches(keypoint_cache_dir, query_image_id, index_image_id,
                        matches0, matches1, matching_scores0, matching_scores1):
    match_fname = f'{keypoint_cache_dir}/match_query_{query_image_id}_index_{index_image_id}.pkl'
    if not ope(match_fname):
        data = {
        'matches0': matches0.cpu().numpy(),
        'matches1': matches1.cpu().numpy(),
        'matching_scores0': matching_scores0.data.cpu().numpy(),
        'matching_scores1': matching_scores1.data.cpu().numpy(),
        }
        with open(match_fname, 'wb') as dbfile:
            pickle.dump(data, dbfile)

def process_resize(w, h, resize):
  assert (len(resize) > 0 and len(resize) <= 2)
  if len(resize) == 1 and resize[0] > -1:
    scale = resize[0] / max(h, w)
    w_new, h_new = int(round(w * scale)), int(round(h * scale))
  elif len(resize) == 1 and resize[0] == -1:
    w_new, h_new = w, h
  else:  # len(resize) == 2:
    w_new, h_new = resize[0], resize[1]

  # Issue warning if resolution is too small or too large.
  if max(w_new, h_new) < 160:
    print('Warning: input resolution is very small, results may vary')
  elif max(w_new, h_new) > 2000:
    print('Warning: input resolution is very large, results may vary')

  return w_new, h_new

def frame2tensor(frame, device='cuda'):
  return torch.from_numpy(frame / 255.).float()[None, None].to(device)

def read_image(path, resize, rotation, resize_float, device='cuda'):
  image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
  if image is None:
    return None, None, None
  w, h = image.shape[1], image.shape[0]
  w_new, h_new = process_resize(w, h, resize)
  scales = (float(w) / float(w_new), float(h) / float(h_new))

  if resize_float:
    image = cv2.resize(image.astype('float32'), (w_new, h_new))
  else:
    image = cv2.resize(image, (w_new, h_new)).astype('float32')

  if rotation != 0:
    image = np.rot90(image, k=rotation)
    if rotation % 2:
      scales = scales[::-1]

  inp = frame2tensor(image, device=device)
  return image, inp, scales
