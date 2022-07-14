import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import scipy.io

import itertools
import torch

DUMMY_MAX = 1e5
epsilon = 1e-4

def get_patches(pc, nn_idx,k):

	batch_size = pc.shape[0]
	num_points = nn_idx.shape[1]

	num_dims = pc.shape[2]

	idx = torch.arange(batch_size) * num_points
	idx = idx.long().reshape((batch_size, 1, 1))
	point_cloud_flat = pc.reshape((-1, num_dims))

	idx = idx.cuda(pc.get_device())

	nnn = (nn_idx + idx).reshape(batch_size * num_points * k)

	point_cloud_neighbors = torch.index_select(point_cloud_flat, 0, index=(nnn).long())

	point_cloud_neighbors_large = point_cloud_neighbors.reshape(batch_size,num_points,k,num_dims)

	return point_cloud_neighbors_large


def pairwise_distance_sqr(point_cloud):

	point_cloud_transpose = point_cloud.permute((0, 2, 1))
	point_cloud_inner = torch.matmul(point_cloud, point_cloud_transpose)
	point_cloud_inner = -2 * point_cloud_inner
	point_cloud_square = torch.sum(torch.pow(point_cloud,2), dim=-1, keepdim=True)
	point_cloud_square_tranpose = point_cloud_square.permute(0, 2, 1)
	return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

def knn(adj_matrix, k=20):
	dist, nn_idx = torch.topk(adj_matrix, k=k, sorted=True, largest = False)

	return dist, nn_idx

class ODFNet_semseg(torch.nn.Module):

	def __init__(self, V):
		super(ODFNet_semseg, self).__init__()

		self.V = V
		self.V_180 = self.V * torch.FloatTensor([-1, -1, 1]).cuda()

		self.direction_neig_count = 32

		self.dist_features = torch.nn.Sequential(
			torch.nn.Linear(6, 32, bias=False),
			torch.nn.BatchNorm1d(32),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(32, 64, bias=False),
			torch.nn.BatchNorm1d(64),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(64, 64, bias=False),
			torch.nn.BatchNorm1d(64),
			torch.nn.LeakyReLU()
		)


		self.cone_features1 = torch.nn.Sequential(
			torch.nn.Linear(8, 32, bias=False),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(32, 16, bias=False),
			torch.nn.LeakyReLU()
		)


		self.cone_folders1 = torch.nn.Sequential(
			torch.nn.Linear(672, 256, bias=False),
			torch.nn.BatchNorm1d(256),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(256, 64, bias=False),
			torch.nn.BatchNorm1d(64),
			torch.nn.LeakyReLU()
		)

		self.cone_pipers1 = torch.nn.Sequential(
			torch.nn.Linear(192, 64, bias=False),
			torch.nn.BatchNorm1d(64),
			torch.nn.LeakyReLU()
		)

		self.cone_pipers2 = torch.nn.Sequential(
			torch.nn.Linear(192, 64, bias=False),
			torch.nn.BatchNorm1d(64),
			torch.nn.LeakyReLU()
		)

		self.cone_pipers3 = torch.nn.Sequential(
			torch.nn.Linear(192, 64, bias=False),
			torch.nn.BatchNorm1d(64),
			torch.nn.LeakyReLU()
		)


		self.total_ODF_conv = torch.nn.Sequential(
			torch.nn.Linear(256, 128, bias=False),
			torch.nn.BatchNorm1d(128),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(128, 128, bias=False),
			torch.nn.BatchNorm1d(128),
			torch.nn.LeakyReLU()
		)


		self.point_features = torch.nn.Sequential(
			torch.nn.Linear(6, 32, bias=False),
			torch.nn.BatchNorm1d(32),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(32, 64, bias=False),
			torch.nn.BatchNorm1d(64),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(64, 128, bias=False),
			torch.nn.BatchNorm1d(128),
			torch.nn.LeakyReLU()
		)

		self.total_conv = torch.nn.Sequential(
			torch.nn.Linear(256, 512, bias=False),
			torch.nn.BatchNorm1d(512),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(512, 1024, bias=False),
			torch.nn.BatchNorm1d(1024),
			torch.nn.LeakyReLU(),
		)

		self.total_conv = torch.nn.Sequential(
			torch.nn.Linear(256, 512, bias=False),
			torch.nn.BatchNorm1d(512),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(512, 1024, bias=False),
			torch.nn.BatchNorm1d(1024),
			torch.nn.LeakyReLU(),
		)

		self.total_conv2 = torch.nn.Sequential(
			torch.nn.Linear(256, 512, bias=False),
			torch.nn.BatchNorm1d(512),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(512, 1024, bias=False),
			torch.nn.BatchNorm1d(1024),
			torch.nn.LeakyReLU(),
		)



		self.outer_linear = torch.nn.Sequential(
			torch.nn.Linear(2048, 1024, bias=False),
			torch.nn.BatchNorm1d(1024),
			torch.nn.LeakyReLU(),
			torch.nn.Dropout(),
			torch.nn.Linear(1024, 512, bias=False),
			torch.nn.BatchNorm1d(512),
			torch.nn.LeakyReLU(),
			torch.nn.Dropout(),
			torch.nn.Linear(512, 256, bias=False),
			torch.nn.BatchNorm1d(256),
			torch.nn.LeakyReLU(),
			torch.nn.Linear(256, 13)
		)

	def get_patch_difference_vectors(self, input, patches):
		batch_size = input.shape[0]
		point_count = input.shape[1]
		patch_size = patches.shape[2]

		tiled_x = input.view((batch_size, 1, point_count, input.shape[2])).repeat((1, patch_size, 1, 1))
		tiled_x = tiled_x.permute((0, 2, 1, 3))

		return (patches - tiled_x).view(-1, point_count, patch_size, input.shape[2])


	def ODF(self, point_cloud, adj_matrix, weight_directions):

		batch_count = point_cloud.shape[0]
		point_count = point_cloud.shape[1]

		distances, nn_idx = knn(adj_matrix, 32 + 1)

		nn_idx = nn_idx[:,:,1:]
		distances = distances[:,:,1:]

		patches = get_patches(point_cloud, nn_idx, 32)

		all_patches = patches[:,:,0:8,:]
		patches = patches[:,:,0:32,:]
		nn_idx = nn_idx[:,:,0:8]

		distance_vectors = self.get_patch_difference_vectors(point_cloud, patches)
		vector_norms = torch.norm(distance_vectors, dim=3).reshape((batch_count, point_count, 32, 1))
		difference_vectors = distance_vectors / (vector_norms)


		weight_directions = weight_directions.reshape((batch_count,point_count,1,self.V.shape[0],3))
		difference_vectors = difference_vectors.reshape((batch_count,point_count,32,1,3))

		angles = torch.mul(weight_directions, difference_vectors)
		angles = torch.sum(angles, 4)


		all_ODFs1 = torch.sign(F.relu(angles - 0.5, inplace=False))


		part1 = (torch.sum(all_ODFs1[:,:,0:8,:],2)/8).unsqueeze(-1)
		part2 = (torch.sum(all_ODFs1[:,:,0:16,:],2)/16).unsqueeze(-1)
		part3 = (torch.sum(all_ODFs1[:,:,0:24,:],2)/24).unsqueeze(-1)
		part4 = (torch.sum(all_ODFs1[:,:,0:32,:],2)/32).unsqueeze(-1)

		all_ODFs2 = torch.sign(F.relu(angles - 0.85065057355, inplace=False))

		part2_1 = (torch.sum(all_ODFs2[:,:,0:8,:],2)/8).unsqueeze(-1)
		part2_2 = (torch.sum(all_ODFs2[:,:,0:16,:],2)/16).unsqueeze(-1)
		part2_3 = (torch.sum(all_ODFs2[:,:,0:24,:],2)/24).unsqueeze(-1)
		part2_4 = (torch.sum(all_ODFs2[:,:,0:32,:],2)/32).unsqueeze(-1)

		all_parts = torch.cat((part1,part2,part3,part4,
							   part2_1, part2_2, part2_3, part2_4, ),3)


		return all_parts, nn_idx, all_patches

	def get_weight_directions(self, point_cloud, adj_matrix):

		batch_count = point_cloud.shape[0]
		point_count = point_cloud.shape[1]

		distances, nn_idx = knn(adj_matrix, self.direction_neig_count + 1)
		patches = get_patches(point_cloud, nn_idx, self.direction_neig_count + 1)[:,:,1:,:]#torch.Size([32, 1024, 32, 3])

		patches_wo_z = patches[:,:,:,0:2] - point_cloud[:,:,0:2].reshape((batch_count,point_count,1,2))

		if self.training:
			rand_border = torch.randint(30,32,(1,))
			mean_xy = torch.mean(patches_wo_z[:,:,0:rand_border,:], 2)
		else:
			mean_xy = torch.mean(patches_wo_z, 2)

		vector_norms = torch.norm(mean_xy, dim=2).unsqueeze(-1)
		mean_xy = mean_xy / (vector_norms + 1e-7)

		mean_xyz = torch.cat((mean_xy, torch.zeros(batch_count,(point_count),1).cuda()),2)

		not_to_be_changed = mean_xyz.reshape((batch_count*(point_count),3))[:,0]

		not_to_be_changed = torch.nonzero(not_to_be_changed > 0.99, as_tuple=False)

		if self.V.shape[0] >= 42:
			V_anchor = self.V[27, :].reshape(1, 1, 3).repeat((batch_count, (point_count), 1))
		else:
			V_anchor = self.V[2, :].reshape(1, 1, 3).repeat((batch_count, (point_count), 1))


		V_anchor_cross = torch.cross(V_anchor, mean_xyz)
		V_cos = torch.mul(mean_xyz, V_anchor)
		V_cos = torch.sum(V_cos, 2).reshape(-1)
		V_anchor_cross = V_anchor_cross.reshape((batch_count * (point_count), 3))
		W = torch.zeros((batch_count * (point_count), 3, 3)).cuda(point_cloud.get_device())
		W[:, 0, 1] = -V_anchor_cross[:, 2]
		W[:, 1, 0] = V_anchor_cross[:, 2]
		W[:, 0, 2] = V_anchor_cross[:, 1]
		W[:, 2, 0] = -V_anchor_cross[:, 1]
		W[:, 1, 2] = -V_anchor_cross[:, 0]
		W[:, 2, 1] = V_anchor_cross[:, 0]
		W_2 = torch.matmul(W, W)
		transform = torch.eye(3).reshape(1, 3, 3).repeat((batch_count * (point_count), 1, 1)).cuda(point_cloud.get_device()) + \
					W + \
					(1 / (1 + V_cos)).reshape((batch_count * (point_count), 1, 1)) * W_2



		V = self.V.reshape((self.V.shape[0], 1, 3, 1))
		weight_directions = torch.matmul(transform, V).squeeze().reshape((self.V.shape[0], batch_count, (point_count), 3))
		weight_directions = weight_directions.permute((1, 2, 0, 3))
		weight_directions = weight_directions.reshape((batch_count*(point_count), self.V.shape[0], 3))

		weight_directions[not_to_be_changed,:,:] = self.V_180

		weight_directions = weight_directions.reshape((batch_count,(point_count), self.V.shape[0], 3))

		return weight_directions

	def get_patch_difference_vectors_with_color(self, points, color, nn_idx):

		input = torch.cat((points,color),2)

		patches = get_patches(input, nn_idx, 8)

		difference_vectors = self.get_patch_difference_vectors(input, patches)

		return difference_vectors


	def forward(self, point_cloud):
		batch_count = point_cloud['points'].shape[0]
		point_count = point_cloud['points'].shape[1]

		adj_matrix = pairwise_distance_sqr(point_cloud['points'])


		oriented_V = self.get_weight_directions(point_cloud['points'], adj_matrix)

		ODFs, nn_idx, patches = self.ODF(point_cloud['points'], adj_matrix, oriented_V)

		distance_vectors = self.get_patch_difference_vectors_with_color(point_cloud['points'], point_cloud['color'], nn_idx)


		ODFs = ODFs.reshape((batch_count*point_count*self.V.shape[0], 8))

		ODFs1 = self.cone_features1(ODFs).reshape((batch_count*point_count, self.V.shape[0] * 16))
		ODFs1 = self.cone_folders1(ODFs1).reshape((batch_count,point_count,64))


		distance_vectors = distance_vectors.reshape((batch_count*point_count*8,6))

		distance_vectors = self.dist_features(distance_vectors).reshape((batch_count, point_count, 8, 64))

		################################

		ODFs1_patches = get_patches(ODFs1, nn_idx, 8)

		ODFs1_repeated = ODFs1.reshape((batch_count,point_count,1,64)).repeat((1, 1, 8, 1))
		ODFs1_patches = torch.cat((ODFs1_repeated, ODFs1_patches, distance_vectors), 3)
		ODFs1_patches = ODFs1_patches.reshape((batch_count*point_count*8, 192))

		ODFs1_patches = self.cone_pipers1(ODFs1_patches).reshape((batch_count,point_count,8,64))


		ODFs2 = torch.max(ODFs1_patches, 2)[0]

		################################

		ODFs2_patches = get_patches(ODFs2, nn_idx, 8)
		ODFs2_repeated = ODFs2.reshape((batch_count,point_count,1,64)).repeat((1, 1, 8, 1))
		ODFs2_patches = torch.cat((ODFs2_repeated, ODFs2_patches, distance_vectors), 3)
		ODFs2_patches = ODFs2_patches.reshape((batch_count*point_count*8, 192))

		ODFs2_patches = self.cone_pipers2(ODFs2_patches).reshape((batch_count,point_count,8,64))

		ODFs3 = torch.max(ODFs2_patches, 2)[0]


		################################

		ODFs3_patches = get_patches(ODFs3, nn_idx, 8)#torch.Size([16, 4096, 8, 64])
		ODFs3_repeated = ODFs3.reshape((batch_count,point_count,1,64)).repeat((1, 1, 8, 1))
		ODFs3_patches = torch.cat((ODFs3_repeated, ODFs3_patches, distance_vectors), 3)
		ODFs3_patches = ODFs3_patches.reshape((batch_count*point_count*8, 192))

		ODFs3_patches = self.cone_pipers3(ODFs3_patches).reshape((batch_count,point_count,8,64))

		ODFs4 = torch.max(ODFs3_patches, 2)[0]


		ODF_features_all = torch.cat((ODFs1,ODFs2,ODFs3,ODFs4),-1).reshape((batch_count*point_count,256))


		ODF_features_all = self.total_ODF_conv(ODF_features_all)

		x = torch.cat((point_cloud['points'], point_cloud['color']), 2).reshape((batch_count*point_count,6))

		point_features_all = self.point_features(x)

		features_all = torch.cat((ODF_features_all,point_features_all),1)

		ODF_features_all = self.total_conv2(features_all)

		pc_rasterized = torch.reshape(ODF_features_all, (batch_count, point_count, 1024))
		out_max = torch.max(pc_rasterized,1)[0]
		out_max = out_max.reshape((batch_count,1, 1024)).repeat((1,point_count,1))

		concat = torch.cat((pc_rasterized, out_max), 2)

		concat = concat.reshape((batch_count*point_count,2048))

		res = self.outer_linear(concat)

		res = res.reshape((batch_count, point_count, 13))

		return res

