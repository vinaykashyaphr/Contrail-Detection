# Contrail-Detection
Based on Kaggle competition "Google Research - Identify Contrails to Reduce Global Warming"

# Project Description
This is a semantic segmentation project, where the contrails captured from GOES-16 satillite has been segmented with different UNet archetectures.
In this project, pytorch-lightning has been used to to train and evaluate the model based on dice score. A maximum of  dice score of 0.645 has been achived.

# Brief Introduction about Dataset
The satellite images are originally obtained from the [GOES-16 Advanced Baseline Imager (ABI)](https://www.goes-r.gov/spacesegment/abi.html), which is publicly available on [Google Cloud Storage](https://console.cloud.google.com/storage/browser/gcp-public-data-goes-16). The original full-disk images were reprojected using bilinear resampling to generate a local scene image. Because contrails are easier to identify with temporal context, a sequence of images at 10-minute intervals are provided. Each example contains exactly one labeled frame.

band_{08-16}.npy: array with size of H x W x T, where T = n_times_before + n_times_after + 1, representing the number of images in the sequence. There are n_times_before and n_times_after images before and after the labeled frame respectively. In our dataset all examples have n_times_before=4 and n_times_after=3. Each band represents an infrared channel at different wavelengths and is converted to brightness temperatures based on the calibration parameters. The number in the filename corresponds to the GOES-16 ABI band number. Details of the ABI bands can be found here.
human_individual_masks.npy: array with size of H x W x 1 x R. Each example is labeled by R individual human labelers. R is not the same for all samples. The labeled masks have value either 0 or 1 and correspond to the (n_times_before+1)-th image in band_{08-16}.npy. They are available only in the training set.
human_pixel_masks.npy: array with size of H x W x 1 containing the binary groundtruth. A pixel is regarded as contrail pixel in evaluation if it is labeled as contrail by more than half of the labelers.

Learn more about the dataset from our preprint: [OpenContrails: Benchmarking Contrail Detection on GOES-16 ABI](https://arxiv.org/abs/2304.02122).

# Model Archetecture:
A modified UNet archetecture with decreased convolution blocks and added convtranspose blocks has been chosen for the segmentation task. However, the various UNET archetectures implemented are availabele in the [unet_arch.py](https://github.com/vinaykashyaphr/Contrail-Detection/blob/main/unet_arch.py) file.

## UNet Variants implemented:
1. UNet 3+
2. Unet with 1 convolution block and 2 convtranspose (to increase pixel wise prediction capacity of the model).
3. Attention UNet
4. UNet from [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Losses implemented:
1. Dice Loss
2. Adaptive Dice Loss
3. Weighted Dice Loss
4. BCE with Dice Loss

# Details
1. Training time - 6.46 hours
2. Std no. of epochs - 15
3. Maximum dice - 0.6457

