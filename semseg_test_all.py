from __future__ import print_function
import os
import torch
from data import S3DISDataLoader
from PC_network import ODFNet_semseg
import numpy as np
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import scipy.io

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def test_sixfold(data_dir):

	test_groups = [0,1,2,3,4,5]
	test_models = ["Model_0_dict.pth","Model_1_dict.pth","Model_2_dict.pth","Model_3_dict.pth","Model_4_dict.pth","Model_5_dict.pth"]

	gt_classes = [0 for _ in range(13)]
	positive_classes = [0 for _ in range(13)]
	true_positive_classes = [0 for _ in range(13)]

	for i in range(len(test_models)):

		exp_group = test_groups[i]
		modelname = "pretrained/S3DIS/"+test_models[i]

		print(exp_group)

		test_loader = DataLoader(S3DISDataLoader('test', data_dir,exp_group), num_workers=1,
								 batch_size=8, shuffle=False, drop_last=False)

		V = torch.from_numpy(scipy.io.loadmat('directions/V_42.mat')['V']).cuda().float()

		model = ODFNet_semseg(V).cuda()
		model.load_state_dict(torch.load(modelname))

		model.eval()
		count = 0

		with torch.no_grad():
			for i, batch in enumerate(test_loader):

				batch_size = batch['points'].shape[0]
				count += batch_size

				batch['points'] = batch['points'].float().cuda()
				batch['color'] = batch['color'].float().cuda()
				batch['label'] = batch['label'].long().cuda()

				logits = model(batch)

				preds = logits.max(dim=2)[1]

				flattened_gt = batch['label'].reshape(-1).cpu().numpy()
				flattened_pred = preds.reshape(-1).cpu().numpy()

				for i in range(13):
					gt_classes[i] += np.sum(1.0*(flattened_gt == i))
					positive_classes[i] += np.sum(1.0*(flattened_pred == i))
					true_positive_classes[i] += np.sum(1.0*(flattened_gt == i)*(flattened_gt == flattened_pred))

	overall_accuraccy = np.sum(true_positive_classes)/np.sum(positive_classes)

	iou_list = np.zeros((13,1))
	for i in range(13):
		iou = true_positive_classes[i] / float(gt_classes[i] + positive_classes[i] - true_positive_classes[i])
		iou_list[i] = iou

	mIoU = np.sum(iou_list)/13.0


	print('mIoU: ' + str(mIoU))
	print('overall_accuraccy: ' + str(overall_accuraccy))


if __name__ == "__main__":

	data_dir = "data/"
	test_sixfold(data_dir)
