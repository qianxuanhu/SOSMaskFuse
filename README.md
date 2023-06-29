# SOSMaskFuse: An Infrared and Visible Image Fusion Architecture Based on Salient Object Segmentation Mask

Li G, Qian X, Qu X. SOSMaskFuse: An Infrared and Visible Image Fusion Architecture Based on Salient Object Segmentation Mask. IEEE Transactions on Intelligent Transportation Systems. 2023 Apr 26. 
DOI: 10.1109/TITS.2023.3268063
- [IEEEXplore](https://ieeexplore.ieee.org/abstract/document/10109138) 

## For test
- #### Model
Make sure the two models provided are on the correct path.
1. The training model "Epoch_model_1.model" of the Autoencoder network. Please will be placed in the model './SOSnetwork/IMV_F_Autoencoder/models/sosmaskfuse_autoencoder/1e2/Epoch_model/'.
2. [Training model of Autoencoder network](https://pan.baidu.com/s/11nz_Cs45VbjIwsOmQeb5AQ?pwd=l22i) (Verification code: l22i.). Please will be placed in the model './SOSnetwork/model/'.
- #### Test images
During the testing of SOSMaskFuse, the folder storing infrared images is named "Inf" and the folder storing visible is named "Vis". (The files in the SOSmask folder are binary significance masks of infrared images generated by the SOSnetwork model.) The corresponding infrared image and visible image file name must be the same. Please place infrared and visible images in the following directory: './SOSnetwork/Test_images/InfVis/'.



## For train
### IMV-F_Autoencoder network
- #### Dataset
[MS-COCO 2014](http://images.cocodataset.org/zips/train2014.zip) is utilized to train our IMV-F_Autoencoder network. [T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 3-5.]
### SOSnetwork
- #### Dataset
[ROAD](https://pan.baidu.com/s/1XmgbVkLgSn9-D83uZRN_DQ?pwd=3txt) (Verification code: 3txt.) is utilized to train our SOSnetwork network. [Takumi, Karasawa, Kohei Watanabe, Qishen Ha, Antonio Tejero-De-Pablos, Yoshitaka Ushiku, and Tatsuya Harada. "Multispectral object detection for autonomous vehicles." In Proceedings of the on Thematic Workshops of ACM Multimedia 2017, pp. 35-43. 2017.]
- #### Model
[hrnetv2_w30_imagenet_pretrained](https://pan.baidu.com/s/1KtDiRjz0hWWGCeQkdOdGhg?pwd=muau) (Verification code: muau.) serves as a pre-training model for training SOS network. Please will be placed in the model './SOSnetwork/model/pretrain_model/'. [Wang J, Sun K, Cheng T, Jiang B, Deng C, Zhao Y, Liu D, Mu Y, Tan M, Wang X, Liu W. Deep high-resolution representation learning for visual recognition. IEEE transactions on pattern analysis and machine intelligence. 2020 Apr 1;43(10):3349-64.]


