#!/usr/bin/env python3

import random
import time
import unittest
import numpy as np

from PIL import Image
import cereal.messaging as messaging
from selfdrive.test.helpers import with_processes
from selfdrive.camerad.snapshot.visionipc import VisionIPC
from common.stat_live import RunningStat

from common.hardware import EON

def _get_snapshots():
  ret = None
  start_time = time.time()
  while time.time() - start_time < 5.0:
    try:
      ipc = VisionIPC()
      pic = ipc.get()
      del ipc

      ipc_front = VisionIPC(front=True)
      fpic = ipc_front.get()
      del ipc_front

      ret = pic, fpic
      break
    except Exception:
      time.sleep(1)
  return ret

def _numpy_bgr2gray(im):
  ret = np.clip(im[:,:,0] * 0.114 + im[:,:,1] * 0.587 + im[:,:,2] * 0.299, 0, 255).astype(np.uint8)
  return ret

def _numpy_lap(im):
  ret = np.zeros(im.shape)
  ret += -4 * im
  ret += np.concatenate([np.zeros((im.shape[0],1)),im[:,:-1]], axis=1)
  ret += np.concatenate([im[:,1:],np.zeros((im.shape[0],1))], axis=1)
  ret += np.concatenate([np.zeros((1,im.shape[1])),im[:-1,:]], axis=0)
  ret += np.concatenate([im[1:,:],np.zeros((1,im.shape[1]))], axis=0)
  ret = np.clip(ret, 0, 255).astype(np.uint8)
  return ret

def _is_really_sharp(i, threshold=800, roi_max=np.array([8,6]), roi_xxyy=np.array([1,6,2,3])):
  i = _numpy_bgr2gray(i)
  x_pitch = i.shape[1] // roi_max[0]
  y_pitch = i.shape[0] // roi_max[1]
  lap = _numpy_lap(i)
  lap_map = np.zeros((roi_max[1], roi_max[0]))
  for r in range(lap_map.shape[0]):
    for c in range(lap_map.shape[1]):
      selected_lap = lap[r*y_pitch:(r+1)*y_pitch, c*x_pitch:(c+1)*x_pitch]
      lap_map[r][c] = 5*selected_lap.var() + selected_lap.max()
  print(lap_map[roi_xxyy[2]:roi_xxyy[3]+1,roi_xxyy[0]:roi_xxyy[1]+1])
  if (lap_map[roi_xxyy[2]:roi_xxyy[3]+1,roi_xxyy[0]:roi_xxyy[1]+1] > threshold).sum() > \
        (roi_xxyy[1]+1-roi_xxyy[0]) * (roi_xxyy[3]+1-roi_xxyy[2]) * 0.9:
    return True
  else:
    return False

def _is_exposure_okay(i, med_ex=np.array([0.2,0.4]), mean_ex=np.array([0.2,0.6])):
  i = _numpy_bgr2gray(i)
  i_median = np.median(i) / 256
  i_mean = np.mean(i) / 256
  print([i_median, i_mean])
  return med_ex[0] < i_median < med_ex[1] and mean_ex[0] < i_mean < mean_ex[1]

def test_camera_operation():
  if EON:
    all_count = 0
    good_count = 0
    bad_count = 0
    sm = messaging.SubMaster(['frame'])
    sharpness_stat = RunningStat(max_trackable=100)
    while True:
      # run checks on AE
      pic, _ = _get_snapshots()
      all_count += 1
      pic_roi = pic[322:322+314,290:290+560]
      ae_good = _is_exposure_okay(pic_roi)
      sharpness_this = 0
      sm.update(0)
      if sm.updated['frame']:
        sharpness_this = np.array(sm['frame'].sharpnessScore).mean()
        sharpness_stat.push_data(sharpness_this)
        print([sharpness_stat.M, sharpness_stat.std()],[sharpness_this])
      sharpness_consistent = True
      if sharpness_stat.n > 50:
        sharpness_consistent = sharpness_stat.M - 2.5*sharpness_stat.std() < sharpness_this <  sharpness_stat.M + 2.5*sharpness_stat.std()
        #print([sharpness_stat.M, sharpness_stat.S],[sharpness_this])
      time.sleep(3)
      bf = (not ae_good) or (not sharpness_consistent)
      if not bf:
        good_count += 1
      else:
        img = Image.fromarray(pic)
        img.save("/data/bf/%d.jpg" % bad_count, "JPEG")
        bad_count += 1
      print("(%d/%d) frame looks good" % (good_count, all_count)) if not bf else print("(%d/%d) HMM FRAME LOOKS WEIRD, saving.." % (bad_count, all_count))
      print("------------------")

if __name__ == "__main__":
  test_camera_operation()
