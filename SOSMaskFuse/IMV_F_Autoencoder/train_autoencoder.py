# -*- coding:utf-8 -*-
#@Project: SOSMaskFuse for image fusion
#@Author: Qian Xuanhu, Shenzhen University
#@Email: qianxuanhu@163.com
#@File : train_autoencoder.py

from args_fusion import args
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_num

import time
import numpy as np
from tqdm import trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from IMV_F_Autoencoder.net import SOSMaskFuse_autoencoder
import pytorch_msssim
import pynvml
from tensorboardX import SummaryWriter


def main():
	original_imgs_path = utils.list_images(args.dataset)
	train_num = 80000
	original_imgs_path = original_imgs_path[:train_num]
	random.shuffle(original_imgs_path)
	i = 2 # i is the index of the coefficient that balances pixel loss and SSIM loss, which defaults to 100 (i = 2).
	train(i, original_imgs_path)


def train(i, original_imgs_path):

	batch_size = args.batch_size
	writer = SummaryWriter('./log')

	# load network model
	input_nc = 1
	output_nc = 1
	nb_filter = [64, 112, 160, 208, 256]

	nest_model = SOSMaskFuse_autoencoder(nb_filter, input_nc, output_nc)

	if args.resume is not None:
		print('Resuming, initializing using weight from {}.'.format(args.resume))
		nest_model.load_state_dict(torch.load(args.resume))
	print(nest_model)

	optimizer = Adam(nest_model.parameters(), args.lr)

	# Loss function
	mse_loss = torch.nn.MSELoss()
	ssim_loss = pytorch_msssim.msssim

	if args.cuda:
		nest_model.cuda()

	tbar = trange(args.epochs)
	print('Start training.....')

	Loss_pixel = []
	Loss_ssim = []
	Loss_all = []

	all_ssim_loss = 0.
	all_pixel_loss = 0.

	for e in tbar:
		print('\nEpoch %d.....' % (e+1))
		# load training database
		image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
		nest_model.train()
		count = 0
		for batch in range(batches):
			image_paths = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
			img = utils.get_train_images_auto(image_paths, height=args.HEIGHT, width=args.WIDTH, flag=False)
			count += 1
			optimizer.zero_grad()
			img = Variable(img, requires_grad=False)
			if args.cuda:
				img = img.cuda()
			# Image multi-scale decomposition and reconstruction.
			# encoder
			en = nest_model.encoder(img)
			# decoder
			outputs = nest_model.decoder_train(en)

			# resolution loss: between the source image(Input) and the reconstructed image(Output)
			x = Variable(img.data.clone(), requires_grad=False)
			ssim_loss_value = 0.
			pixel_loss_value = 0.
			# pixel_loss
			pixel_loss_temp = mse_loss(outputs, x)
			pixel_loss_value += pixel_loss_temp
			# ssim_loss: The more similar the structure, the greater the value and the maximum is 1.
			ssim_loss_temp = ssim_loss(outputs, x, normalize=True)
			ssim_loss_value += (1-ssim_loss_temp)

			# total loss
			# ssim_weight = [1, 10, 100, 1000, 10000]
			total_loss = pixel_loss_value + args.ssim_weight[i] * ssim_loss_value

			total_loss.backward()
			optimizer.step()

			writer.add_scalar('baseline_loss/Loss_pixel', pixel_loss_value, batches * e + batch + 1)
			writer.add_scalar('baseline_loss/Loss_ssim', ssim_loss_value, batches * e + batch + 1)
			writer.add_scalar('baseline_loss/Loss_all', total_loss, batches * e + batch + 1)

			all_ssim_loss += ssim_loss_value.item()
			all_pixel_loss += pixel_loss_value.item()

			if np.isnan(all_pixel_loss):
				print('Pixel loss error')
				exit()
			if np.isnan(all_ssim_loss):
				print('SSIM loss error')
				exit()

			if (batch + 1) % args.log_interval == 0:
				if args.cuda:
					pynvml.nvmlInit()
					handle = pynvml.nvmlDeviceGetHandleByIndex(int(args.GPU_num)) # GPU_num
					meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
					mesg = "{} 'cuda:{}': [{:.1f}/{:.1f}G]\t ssim_weight {}\t " \
						   "Epoch {}: [{}/{}]\t pixel_loss: {:.6f}\t ssim_loss: {:.6f}\t total: " \
						   "{:.6f}".format(time.ctime(), args.GPU_num, meminfo.used/1024**3, meminfo.total/1024**3, i,
										   e + 1, count, batches, all_pixel_loss / args.log_interval,
										 (args.ssim_weight[i] * all_ssim_loss) / args.log_interval,
										 (args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval)
				else:
					mesg = "{}\t part_ssim weight {}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: " \
						   "{:.6f}\t total: {:.6f}".format(time.ctime(), i, e + 1, count, batches,
									  all_pixel_loss / args.log_interval,
									  (args.ssim_weight[i] * all_ssim_loss) / args.log_interval,
									  (args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval)

				tbar.set_description(mesg)
				Loss_pixel.append(all_pixel_loss / args.log_interval)
				Loss_ssim.append(all_ssim_loss / args.log_interval)
				Loss_all.append((args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval)
				all_ssim_loss = 0.
				all_pixel_loss = 0.
			writer.close()

			if (batch + 1) % (200 * args.log_interval) == 0:
				# save model
				nest_model.eval()
				nest_model.cpu()
				save_model_filename = args.ssim_path[i] + '/' + "Epoch_" + str(e + 1) + "_iters_" + str(count) + "_" + \
									  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] \
									  + ".model"
				# Creating a folder
				if not os.path.exists(args.save_model_dir_autoencoder):
					os.makedirs(args.save_model_dir_autoencoder)
				path_model = os.path.join(args.save_model_dir_autoencoder, args.ssim_path[i])

				if not os.path.exists(path_model):
					os.makedirs(path_model)

				save_model_path = os.path.join(args.save_model_dir_autoencoder, save_model_filename)
				torch.save(nest_model.state_dict(), save_model_path)

				# save loss data
				# Creating a folder
				if not os.path.exists(args.save_loss_dir):
					os.makedirs(args.save_loss_dir)
				path_lossdata = os.path.join(args.save_loss_dir, args.ssim_path[i])
				if not os.path.exists(path_lossdata):
					os.makedirs(path_lossdata)

				# pixel loss
				loss_data_pixel = Loss_pixel
				loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_pixel_epoch_" + \
									 str(e + 1) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_')\
										 .replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
				scio.savemat(loss_filename_path, {'loss_pixel': loss_data_pixel})
				# SSIM loss
				loss_data_ssim = Loss_ssim
				loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_ssim_epoch_" + \
									 str(e + 1) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_')\
										 .replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
				scio.savemat(loss_filename_path, {'loss_ssim': loss_data_ssim})
				# all loss
				loss_data = Loss_all
				loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_all_epoch_" + \
									 str(e + 1) + "_iters_" + str(count) + "-" + str(time.ctime()).replace(' ', '_')\
										 .replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
				scio.savemat(loss_filename_path, {'loss_all': loss_data})

				nest_model.train()
				nest_model.cuda()
				tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

		#Each epoch save model is used to fuse and view evaluation metrics
		nest_model.eval()
		nest_model.cpu()
		epoch_model_path = args.ssim_path[i] + '/Epoch_model'
		path_epoch_model = os.path.join(args.save_model_dir_autoencoder, epoch_model_path)
		# Creating a folder
		if not os.path.exists(path_epoch_model):
			os.makedirs(path_epoch_model)
		save_epoch_model_name = epoch_model_path + '/' + "Epoch_model" + "_" + str(e+1) + ".model"

		save_epoch_model_path = os.path.join(args.save_model_dir_autoencoder, save_epoch_model_name)
		torch.save(nest_model.state_dict(), save_epoch_model_path)

		nest_model.train()
		nest_model.cuda()

	# Save final data
	# pixel loss
	loss_data_pixel = Loss_pixel
	loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_pixel_epoch_" + str(
		args.epochs) + "_" + str(
		time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
	scio.savemat(loss_filename_path, {'final_loss_pixel': loss_data_pixel})
	# SSIM loss
	loss_data_ssim = Loss_ssim
	loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_ssim_epoch_" + str(
		args.epochs) + "_" + str(
		time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
	scio.savemat(loss_filename_path, {'final_loss_ssim': loss_data_ssim})
	# all loss
	loss_data = Loss_all
	loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_all_epoch_" + str(
		args.epochs) + "_" + str(
		time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
	scio.savemat(loss_filename_path, {'final_loss_all': loss_data})
	# save model
	nest_model.eval()
	nest_model.cpu()
	save_model_filename = args.ssim_path[i] + '/' "Final_epoch_" + str(args.epochs) + "_" + \
						  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".model"
	save_model_path = os.path.join(args.save_model_dir_autoencoder, save_model_filename)
	torch.save(nest_model.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)


if __name__ == "__main__":
	main()
