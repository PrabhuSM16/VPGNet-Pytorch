# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score

def create_circular_mask(h, w, center=None, radius=None):
  # Create a circular mask from center on an (h,w) map with euclidean distance radius
  if center is None: # use the middle of the image
    center = (int(w/2), int(h/2))
  if radius is None: # use the smallest distance between the center and image walls
    radius = min(center[0], center[1], w-center[0], h-center[1])

  Y, X = np.ogrid[:h, :w]
  dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

  mask = dist_from_center <= radius
  return mask

def compute_class_scores(pred, gt, classes, mask_R=4):
  # Use original tensor shape and normalized values from net output
  # Returns accumulated scores for batch
  pred_shape = pred.shape
  bsize, ncls, h, w = pred_shape
  assert pred_shape == gt.shape, f'Error: Pred {pred_shape} and GT {gt.shape} have mismatched shapes!'
  
  # Generate empty accumulator for all scores
  f1_dict = {i+1: 0 for i in range(len(classes))}
  recall_dict = {i+1: 0 for i in range(len(classes))}
  precision_dict = {i+1: 0 for i in range(len(classes))}
  
  for b in range(bsize):
    # per image loop
    for c in classes:
      # per image per class loop      
      pred_mask = pred[b,c-1,:,:].cpu().numpy() > 0
      gt_mask = gt[b,c-1,:,:].cpu().numpy() > 0
      
      if 'lane' in classes[c]:
        # only identified lane classes
        extend_mask = np.ones((h,w), dtype=bool) # extended groundtruth (from 8*8 square grid to radius R circle)
        # iterate across map
        for i in range(h):
          for j in range(w):
            if gt_mask[i,j] == True: # if this pixel have label, this 8*8 grid should have same label
              area_mask = create_circular_mask(h, w, center=(i,j), radius=mask_R)
              extend_mask = extend_mask + area_mask # add the area_mask to blank mask
        
        # Compare pred and the extended mask (gt) for f1 score
        f1_dict[c] += f1_score(extend_mask.flatten(), pred_mask.flatten())
        recall_dict[c] += recall_score(extend_mask.flatten(), pred_mask.flatten())
        precision_dict[c] += precision_score(extend_mask.flatten(), pred_mask.flatten())
      
      else:
        # all other roadmarker classes
        f1_dict[c] += f1_score(gt_mask.flatten(), pred_mask.flatten())
        recall_dict[c] += recall_score(gt_mask.flatten(), pred_mask.flatten())
        precision_dict[c] += precision_score(gt_mask.flatten(), pred_mask.flatten())
  
  # take average across batch
  f1_dict = {classes[c]: f1_dict[c]/bsize for c in f1_dict}
  recall_dict = {classes[c]: recall_dict[c]/bsize for c in recall_dict}
  precision_dict = {classes[c]: precision_dict[c]/bsize for c in precision_dict}
  return f1_dict, recall_dict, precision_dict

def compute_vp_scores(pred, gt):
  # Use original tensor shape and normalized values from net output
  # Returns accumulated scores for batch
  f1, recall, precision = 0, 0, 0
  pred_shape = pred.shape
  bsize, h, w = pred_shape
  assert pred_shape == gt.shape, f'Error: Pred {pred_shape} and GT {gt.shape} have mismatched shapes!'
  
  for b in range(bsize):
    # individual batch size
    pred_mask = pred[b,:,:].cpu().numpy() > 0
    gt_mask = gt[b,:,:].cpu().numpy() > 0
    # compute and accumulate scores across batch
    f1 += f1_score(gt_mask.flatten(), pred_mask.flatten())
    recall += recall_score(gt_mask.flatten(), pred_mask.flatten())
    ##precision += precision_score(gt_mask.flatten(), pred_mask.flatten())
  
  # take average across batch
  f1 /= bsize
  recall /= bsize
  #precision /= bsize
  return f1, recall, precision
