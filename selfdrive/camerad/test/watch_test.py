#!/usr/bin/env python3

import random
import time
import unittest
import numpy as np

import cereal.messaging as messaging
from selfdrive.test.helpers import with_processes
from selfdrive.camerad.snapshot.visionipc import VisionIPC

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
  i = self._numpy_bgr2gray(i)
  x_pitch = i.shape[1] // roi_max[0]
  y_pitch = i.shape[0] // roi_max[1]
  lap = self._numpy_lap(i)
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
  i = self._numpy_bgr2gray(i)
  i_median = np.median(i) / 256
  i_mean = np.mean(i) / 256
  print([i_median, i_mean])
  return med_ex[0] < i_median < med_ex[1] and mean_ex[0] < i_mean < mean_ex[1]

def test_camera_operation():
  print("checking image outputs")
  if EON:
    # run checks similar to prov
    time.sleep(5) # wait for startup and AF
    pic, _ = _get_snapshots()
    assert _is_really_sharp(pic)
    assert _is_exposure_okay(pic)

if __name__ == "__main__":
  test_camera_operation()
