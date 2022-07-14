import torch
import torch.utils.data
import numpy as np
import os
import h5py
import scipy.io
epsilon = 1e-4


def rotate_point_cloud(batch_data):

	rand_val = np.random.randint(5)

	rotation_angle = rand_val * np.pi /2
	cosval = np.cos(rotation_angle)
	sinval = np.sin(rotation_angle)
	rotation_matrix = np.array([[cosval, sinval, 0],
								[-sinval, cosval, 0],
								[0, 0, 1]])

	rotated_data = np.dot(batch_data, rotation_matrix)

	return rotated_data

class S3DISDataLoader(torch.utils.data.Dataset):

	def __init__(self,mode, main_dir, sel_room):

		self.mode = mode

		self.main_dir = main_dir
		ALL_FILES = [line.rstrip() for line in
					 open(os.path.join(self.main_dir, 'indoor3d_sem_seg_hdf5_data/all_files.txt'))]

		room_filelist = [line.rstrip() for line in
						 open(os.path.join(self.main_dir, 'indoor3d_sem_seg_hdf5_data/room_filelist.txt'))]

		data_batch_list = []
		label_batch_list = []

		for h5_filename in ALL_FILES:
			f = h5py.File(os.path.join(self.main_dir, h5_filename))
			data_batch = f['data'][:]
			label_batch = f['label'][:]
			data_batch_list.append(data_batch)
			label_batch_list.append(label_batch)

		data_batches = np.concatenate(data_batch_list, 0)
		label_batches = np.concatenate(label_batch_list, 0)

		test_area = 'Area_' + str(sel_room+1)
		train_idxs = []
		test_idxs = []

		for i, room_name in enumerate(room_filelist):
			if test_area in room_name:
				test_idxs.append(i)
			else:
				train_idxs.append(i)

		if mode == 'train':
			self.data = data_batches[train_idxs, ...]
			self.label = label_batches[train_idxs]
		else:
			self.data = data_batches[test_idxs, ...]
			self.label = label_batches[test_idxs]




	def __getitem__(self, index):

		sel_data = self.data[index,:,:]

		selected_item = {}

		selected_item['points'] = sel_data[:,0:3]
		selected_item['color'] = sel_data[:,3:6]
		selected_item['label'] = self.label[index,:]

		if self.mode == 'train':

			selected_item['points'] = rotate_point_cloud(selected_item['points'])

			if np.random.random() < 0.5:
				selected_item['points'][:, 0] = -selected_item['points'][:, 0]

			if np.random.random() < 0.5:
				selected_item['points'][:, 1] = -selected_item['points'][:, 1]

		return selected_item

	def __len__(self):
		return self.data.shape[0]