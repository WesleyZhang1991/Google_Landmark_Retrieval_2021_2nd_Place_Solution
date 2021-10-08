# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import torch
import time

from .superpoint import SuperPoint
from .superglue import SuperGlue


class Matching(torch.nn.Module):
  """ Image Matching Frontend (SuperPoint + SuperGlue) """
  def __init__(self, config={}):
    super().__init__()
    self.superpoint = SuperPoint(config.get('superpoint', {}))
    self.superglue = SuperGlue(config.get('superglue', {}))

  def do_superpoint(self, data):
    pred = {}

    # Extract SuperPoint (keypoints, scores, descriptors) if not provided
    if 'keypoints0' not in data:
      pred0 = self.superpoint({'image': data['image0']})
      pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
    else:
      pred = {**pred, **{'keypoints0': data['keypoints0'], 'scores0': data['scores0'], 'descriptors0': data['descriptors0']}}
      data.pop('keypoints0')
      data.pop('scores0')
      data.pop('descriptors0')
    if 'keypoints1' not in data:
      pred1 = self.superpoint({'image': data['image1']})
      pred = {**pred, **{k+'1': v for k, v in pred1.items()}}
    else:
      pred = {**pred, **{'keypoints1': data['keypoints1'], 'scores1': data['scores1'], 'descriptors1': data['descriptors1']}}
      data.pop('keypoints1')
      data.pop('scores1')
      data.pop('descriptors1')

    return pred

  def do_superglue(self, data, pred):
    # Batch all features
    # We should either have i) one image per batch, or
    # ii) the same number of local features for all images in the batch.
    data = {**data, **pred}

    for k in data:
      if isinstance(data[k], (list, tuple)):
        data[k] = torch.stack(data[k])

    # Perform the matching
    # print(f'data : {data.keys()}')
    pred = {**pred, **self.superglue(data)}

    return pred

  def forward(self, data):
    """ Run SuperPoint (optionally) and SuperGlue
    SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
    Args:
      data: dictionary with minimal keys: ['image0', 'image1']
    """
    start = time.time()
    pred = self.do_superpoint(data)
    extract = time.time()
    extract_time = extract - start
    pred = self.do_superglue(data, pred)
    matching = time.time()
    matching_time = matching - extract
    return pred, extract_time, matching_time
