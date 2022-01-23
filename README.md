# HistoSeg : Quick attention with multi-loss function for multi-structure segmentation in digital histology images

<a href="#"><img src="https://img.shields.io/badge/Maintained-Yes-2ea44f?style=for-the-badge" alt="Maintained - Yes"></a>
<a href="#"><img src="https://img.shields.io/badge/Quick_Attention-6fa8dc?style=for-the-badge" alt="Quick Attention"></a>
<a href="#"><img src="https://img.shields.io/badge/Multi_Loss_Function-74867C?style=for-the-badge" alt="Multi Loss Function"></a>
<a href="#"><img src="https://img.shields.io/badge/Encoder--Decoder_Network-A2AF48?style=for-the-badge" alt="Encoder-Decoder Network"></a>
<a href="#"><img src="https://img.shields.io/badge/Semantic_Segmentation-b6d7a8?style=for-the-badge" alt="Semantic Segmentation"></a>
<a href="#"><img src="https://img.shields.io/badge/Computational_Pathology-e06666?style=for-the-badge" alt="Computational Pathology"></a>
<a href="#"><img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" /></a>
<a href="#"><img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white" /></a>
<a href="#"><img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen" /></a>

Histological Image Segmentation<br>This repo contains the code to Test and Train the HistoSeg <br>

HistoSeg is an Encoder-Decoder DCNN which utilizes the novel Quick Attention Modules and Multi Loss function to generate segmentation masks from histopathological images with greater accuracy.


## Datasets used for trainig HistoSeg

### MoNuSeg - Multi-organ nuclei segmentation from H&E stained histopathological images
link: https://monuseg.grand-challenge.org/

### GlaS -  Gland segmentation in histology images
link: https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/

## Trained Weights are available in the repo to test the HistoSeg
For MoNuSeg Dataset link: https://github.com/saadwazir/HistoSeg/blob/main/HistoSeg_MoNuSeg_.h5

For GlaS Dataset link: https://github.com/saadwazir/HistoSeg/blob/main/HistoSeg_GlaS_.h5

## Data Preprocessing for Training
After downloading the dataset you must generate patches of images and their corresponding masks (Ground Truth), & convert it into numpy arrays or you can use dataloaders directly inside the code.
you can generate patches using Image_Patchyfy. Link : https://github.com/saadwazir/Image_Patchyfy
```
For example to train HistoSeg on MoNuSeg Dataset, the distribution of dataset after creating pathes

X_train 1470x256x256x3 
y_train 1470x256x256x1
X_val 686x256x256x3
y_Val 686x256x256x1
```
## Data Preprocessing for Testing
You just need to resize the images and their corresponding masks (Ground Truth) into same size i.e all the samples must have same resolution, and then convert it into numpy arrays.

```
For example to test HistoSeg on MoNuSeg Dataset, the shapes of dataset after creating numpy arrays are

X_test 14x1000x1000x3 
y_test 14x1000x1000x1
```

## Requirements
```pip install scikit-image
pip install matplotlib
pip install seaborn
pip install tqdm
pip install scikit-learn
conda install tensorflow==2.7
pip install keras==2.2.4
```


## Training
To train HistoSeg use the following command

```
python HistoSeg_Train.py --train_images 'path' --train_masks 'path' --val_images 'path' --val_masks 'path' --width 256 --height 256 --epochs 100 --batch 16
```
## Testing
To test HistoSeg use the following command
```
python HistoSeg_Test.py --images 'path' --masks 'path' --weights 'path' --width 1000 --height 1000

For example to test HistoSeg on MoNuSeg Dataset with trained weights, use the following command
python HistoSeg_Test.py --images 'X_test_MoNuSeg_14x1000x1000.npy' --masks 'y_test_MoNuSeg_14x1000x1000.npy' --weights 'HistoSeg_MoNuSeg_.h5' --width 1000 --height 1000
```
