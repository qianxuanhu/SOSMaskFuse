# -*- coding:utf-8 -*-
#@Project: SOSMaskFuse for image fusion
#@Author: Qian Xuanhu, Shenzhen University
#@Email: qianxuanhu@163.com
#@File : run.py

import time
from SOSnetwork import SOSnetwork_test
from IMV_F_Autoencoder import IMV_F_test

if __name__ == '__main__':
	time_s = time.time()
	# Test the path where the image folder is stored. This path should contain two folders: "Inf" and "Vis".
	test_path = './Test_images/InfVis/'

	# The SOS Mask output folder path. The output is sent to the test_path for subsequent use in the IMV-F fusion strategy.
	SOSMask_out_path_root = './Test_images/InfVis/'

	# The path of SOS network model.
	SOSnetwork_model_path = './SOSnetwork/model/normal_epoch_170.pth'

	# The significance mask of infrared image is segmented by SOS network.
	SOSnetwork_test.SOSMask_run(test_path, SOSMask_out_path_root, SOSnetwork_model_path)

	# Infrared, salient mask and visible light images were input into IMV-F fusion strategy for image fusion.
	IMV_F_test.fuse_main(0, 2, test_path)

	time_e = time.time()
	print('Total time: %f s' % (time_e - time_s))