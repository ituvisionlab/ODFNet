from __future__ import print_function
import os
import argparse
import torch
import torch.optim as optim
from data import S3DISDataLoader
from PC_network import ODFNet_semseg
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
NUM_CATEGORIES = 16



def weight_init(m):
	if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
		torch.nn.init.kaiming_uniform(m.weight)



def train(args):

	print('Training for group ' + str(args.exp_group))


	train_loader = DataLoader(S3DISDataLoader('train', "data/", args.exp_group), num_workers=8,
							  batch_size=args.batch_size, shuffle=True, drop_last=True)
	test_loader = DataLoader(S3DISDataLoader('test', "data/",args.exp_group), num_workers=1,
							 batch_size=8, shuffle=True, drop_last=False)

	all_tr_losses = np.zeros((args.epochs, 1))
	all_tr_accuracies = np.zeros((args.epochs, 1))
	all_test_losses = np.zeros((args.epochs, 1))
	all_test_mIoU = np.zeros((args.epochs, 1))
	all_test_overall = np.zeros((args.epochs, 1))

	if args.dir_count == 162:
		V = torch.from_numpy(scipy.io.loadmat('directions/V_162.mat')['V']).cuda().float()
	elif args.dir_count == 12:
		V = torch.from_numpy(scipy.io.loadmat('directions/V_12.mat')['V']).cuda().float()
	else:
		V = torch.from_numpy(scipy.io.loadmat('directions/V_42.mat')['V']).cuda().float()

	model = ODFNet_semseg(V).cuda()
	model = model.float()
	model.apply(weight_init)


	opt = optim.Adam(model.parameters(), lr=0.001)

	criterion = cal_loss

	best_test_acc = 0
	for epoch in range(args.epochs):

		if epoch in [75, 150, 225]:
			for param_group in opt.param_groups:
				param_group['lr'] = param_group['lr']

		####################
		# Train
		####################
		total_loss = 0.0
		model.train()

		selected_loader = train_loader

		count = 0
		total_true = 0
		total_false = 0

		for i, batch in enumerate(selected_loader):


			batch_size = batch['points'].shape[0]
			count += 1


			batch['points'] = batch['points'].float().cuda()
			batch['color'] = batch['color'].float().cuda()
			batch['label'] = batch['label'].long().cuda()

			opt.zero_grad()

			logits = model(batch)

			loss_trial = criterion(logits, batch['label'])

			loss_trial.backward()
			opt.step()

			preds = logits.max(dim=2)[1]

			batch_true_count = torch.sum(1.0*(preds == batch['label'])).item()
			batch_false_count = torch.sum(1.0*(preds != batch['label'])).item()

			total_true  += batch_true_count
			total_false += batch_false_count
			total_loss += loss_trial.item()

			if i % 5 == 0:
				print('Train '+str(i))
				print('Training Loss: %f' % (total_loss * 1.0 / (count)))
				print('Training Acc: %f' % (total_true * 1.0 / (total_false + total_true)))

		total_loss = total_loss * 1.0 / (count)
		total_acc = (total_true * 1.0 / (total_false + total_true))

		print('Total Train ' + str(epoch))
		print('Training Total Mean_loss: %f' % total_loss)
		print('Training Seg Accuracy: %f' % total_acc)

		all_tr_losses[epoch] = total_loss*1.0
		all_tr_accuracies[epoch] = total_acc


		####################
		# Test
		####################

		gt_classes = [0 for _ in range(13)]
		positive_classes = [0 for _ in range(13)]
		true_positive_classes = [0 for _ in range(13)]
		total_loss = 0.0
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
				loss_trial = criterion(logits, batch['label'])
				total_loss += loss_trial.item()

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

		all_test_mIoU[epoch] = mIoU
		all_test_overall[epoch] = overall_accuraccy
		all_test_losses[epoch] = total_loss

		np.save('TR_Losses.npy', all_tr_losses)
		np.save('TS_Losses.npy', all_test_losses)
		np.save('TR_Acc.npy', all_tr_accuracies)
		np.save('TS_mIoU.npy', all_test_mIoU)
		np.save('TS_Acc.npy', all_test_overall)
		torch.save(model, 'Model_'+str(epoch)+'.pth')

		if overall_accuraccy >= best_test_acc:
			best_test_acc = overall_accuraccy
			torch.save(model, 'Model_' + str(epoch) + '_best.pth')

		plt.xlim(xmax=epoch + 1, xmin=0)
		plt.plot(range(0, epoch + 1), all_tr_accuracies[0:epoch + 1])
		plt.ylabel('TrainAccuracies')
		plt.savefig("TrainAccuracies.png")
		plt.clf()
		plt.close()

		plt.xlim(xmax=epoch + 1, xmin=0)
		plt.plot(range(0, epoch + 1), all_test_mIoU[0:epoch + 1])
		plt.ylabel('all_test_mIoU')
		plt.savefig("all_test_mIoU.png")
		plt.clf()
		plt.close()

		plt.xlim(xmax=epoch + 1, xmin=0)
		plt.plot(range(0, epoch + 1), all_test_overall[0:epoch + 1])
		plt.ylabel('all_test_overall')
		plt.savefig("all_test_overall.png")
		plt.clf()
		plt.close()


		plt.xlim(xmax=epoch + 1, xmin=0)
		plt.plot(range(0, epoch + 1), all_tr_losses[0:epoch + 1])
		plt.ylabel('TRlosses')
		plt.savefig("TRLosses.png")
		plt.clf()
		plt.close()

		plt.xlim(xmax=epoch + 1, xmin=0)
		plt.plot(range(0, epoch + 1), all_test_losses[0:epoch + 1])
		plt.ylabel('Vallosses')
		plt.savefig("ValLosses.png")
		plt.clf()
		plt.close()



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp_group', type=int, default=0)
	parser.add_argument('--batch_size', type=int, default=4)
	parser.add_argument('--epochs', type=int, default=250)
	parser.add_argument('--dir_count', type=int, default=12, choices=[12,42,162])



	args = parser.parse_args()

	train(args)
