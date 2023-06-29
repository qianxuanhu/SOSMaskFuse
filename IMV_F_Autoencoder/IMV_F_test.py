# -*- coding:utf-8 -*-
#@Project: SOSMaskFuse for image fusion
#@Author: Qian Xuanhu, Shenzhen University
#@Email: qianxuanhu@163.com
#@File : IMV_F_test.py

import os
import torch
from torch.autograd import Variable
from IMV_F_Autoencoder.net import SOSMaskFuse_autoencoder
from IMV_F_Autoencoder import utils
from IMV_F_Autoencoder.fusion_strategy import IMV_F
from IMV_F_Autoencoder.args_fusion import args
import numpy as np


def load_model(path):
	input_nc = 1
	output_nc = 1
	nb_filter = [64, 112, 160, 208, 256]

	nest_model = SOSMaskFuse_autoencoder(nb_filter, input_nc, output_nc)
	nest_model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('\nModel {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

	nest_model.eval()

	if args.cuda:
		nest_model.cuda()

	return nest_model


def run_demo(nest_model, infrared_path, visible_path, output_path_root, file_name, inf_mask_path):
	img_ir = utils.get_test_image(infrared_path)
	img_vi = utils.get_test_image(visible_path)
	img_ir_mask = utils.get_test_image(inf_mask_path)
	img_ir_mask = img_ir_mask/255 # Binary mask, the value range is [0,1]

	if args.cuda:
		img_ir = img_ir.cuda()
		img_vi = img_vi.cuda()
		img_ir_mask = img_ir_mask.cuda()

	img_ir = Variable(img_ir, requires_grad=False)
	img_vi = Variable(img_vi, requires_grad=False)
	img_ir_mask = Variable(img_ir_mask, requires_grad=False)

	inf_vis_salient_background = IMV_F().Inf_Vis_Salient_Background

	img_ir_ahead, img_ir_back, img_vi_ahead, img_vi_back = inf_vis_salient_background(img_ir, img_vi, img_ir_mask)

	en_r_ahead = nest_model.encoder(img_ir_ahead)
	en_r_back = nest_model.encoder(img_ir_back)

	en_v_ahead = nest_model.encoder(img_vi_ahead)
	en_v_back = nest_model.encoder(img_vi_back)

	# fusion
	f = nest_model.IMV_F_fusion_strategy(en_r_ahead, en_r_back, en_v_ahead, en_v_back)
	# decoder
	img_fusion = nest_model.decoder_eval(f)

	output_path = output_path_root + file_name
	# save images
	utils.save_image_test(img_fusion, output_path)
	# print(output_path)


def fuse_main(epoch, ssim_path_id, test_path):
	# run demo
	test_path = test_path + 'Inf' # test_path = './images/InfVis/Inf/'
	file_name = os.listdir(test_path)

	model_path_file = args.model_default + args.ssim_path[ssim_path_id] + '/Epoch_model/'


	epoch_num = epoch + 1
	model_name = 'Epoch_model_' + str(epoch_num) + '.model'
	epoch_model_path = os.path.join(model_path_file, model_name)
	with torch.no_grad():
		model_path = epoch_model_path
		model = load_model(model_path)

		# Create a folder for epoch
		output_path = './outputs/'
		# output_path = '../outputs/' + 'Epoch_' + str(epoch_num) #if need
		if os.path.exists(output_path) is False:
			os.mkdir(output_path)

		output_path = output_path + '/'
		print('Processing......fuse...  ' + 'epoch_' + str(epoch_num))

		for i in range(len(file_name)):
			infrared_path = os.path.join(test_path, file_name[i])
			visible_path_, ir_mask_path_ = test_path.split('/'), test_path.split('/')
			if visible_path_[-1] is '':
				visible_path_[-2] = 'Vis'
				ir_mask_path_[-2] = 'SOSmask'
			else:
				visible_path_[-1] = 'Vis'
				ir_mask_path_[-1] = 'SOSmask'
			visible_path = os.path.join('/'.join(visible_path_), file_name[i])
			ir_mask_path = os.path.join('/'.join(ir_mask_path_), file_name[i])

			run_demo(model, infrared_path, visible_path, output_path, file_name[i], ir_mask_path)

	print('Done......')


if __name__ == '__main__':
	test_path = '../Test_images/InfVis/Inf/'
	fuse_main(0, 2, test_path)
