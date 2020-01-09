import numpy as np
import itertools

fm_sizes = [38, 19, 10, 5, 3, 1]
scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
aspect_ratios = [[1, 2, 0.5], [1, 2, 3, 0.5, 0.333],
				 [1, 2, 3, 0.5, 0.333], [1, 2, 0.5],
				 [1, 2, 0.5], [1, 2, 0.5]]
offset = 0.5

def generate_prior(fm_id):
	prior_boxes = []
	fm_size = fm_sizes[fm_id]
	scale = scales[fm_id]
	for row, col in itertools.product(range(fm_size), repeat=2):
		cx = (col + offset) / fm_size
		cy = (row + offset) / fm_size

		for ratio in aspect_ratios[fm_id]:
			w = scale * np.sqrt(ratio)
			h = scale / np.sqrt(ratio)
			prior_boxes.append([cx, cy, w, h])
			if ratio == 1:
				try:
					additional_scale = np.sqrt(scale * scales[fm_id + 1])
				except IndexError:
					additional_scale = 1
				prior_boxes.append([cx, cy, additional_scale, additional_scale])
	prior_boxes = np.clip(prior_boxes, 0, 1)
	return prior_boxes
