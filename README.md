<a href="#"><img src="https://img.shields.io/badge/Author_%2F_Corresponding_Person%3A-8e7cc3" alt="Author / Corresponding Person:"></a> <br>
Saad Wazir [saadwazir.pk@gmail.com]

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/histoseg-quick-attention-with-multi-loss/medical-image-segmentation-on-glas)](https://paperswithcode.com/sota/medical-image-segmentation-on-glas?p=histoseg-quick-attention-with-multi-loss)

<a href="https://github.com/saadwazir/HistoSeg"><img src="https://img.shields.io/badge/HistoSeg-Quick_attention_with_multi--loss_function_for_multi--structure_segmentation_in_digital_histology_images-FFC300?style=for-the-badge" alt="HistoSeg - Quick attention with multi-loss function for multi-structure segmentation in digital histology images"></a>
<a href="#"><img src="https://img.shields.io/badge/Maintained-Yes-2ea44f?style=for-the-badge" alt="Maintained - Yes"></a>
<a href="#"><img src="https://img.shields.io/badge/Quick_Attention-6fa8dc?style=for-the-badge" alt="Quick Attention"></a>
<a href="#"><img src="https://img.shields.io/badge/Multi_Loss_Function-74867C?style=for-the-badge" alt="Multi Loss Function"></a>
<a href="#"><img src="https://img.shields.io/badge/Encoder--Decoder_Network-A2AF48?style=for-the-badge" alt="Encoder-Decoder Network"></a>
<a href="#"><img src="https://img.shields.io/badge/Semantic_Segmentation-b6d7a8?style=for-the-badge" alt="Semantic Segmentation"></a>
<a href="#"><img src="https://img.shields.io/badge/Computational_Pathology-e06666?style=for-the-badge" alt="Computational Pathology"></a>
<a href="#"><img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" /></a>
<a href="#"><img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white" /></a>
<a href="#"><img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen" /></a>

# HistoSeg : Quick attention with multi-loss function for multi-structure segmentation in digital histology images

Paper was presented at 12th International Conference on Pattern Recognition Systems (ICPRS), 2022
École Nationale Supérieure des Mines de Saint-Étienne, France

<a href="https://www.researchgate.net/publication/362817207_HistoSeg_Quick_attention_with_multi-loss_function_for_multi-structure_segmentation_in_digital_histology_images">Download Paper</a>

DOI: 10.1109/ICPRS54038.2022.9854067

Copyrights has been given to IEEE. IEEE Xplore link is <a href="https://ieeexplore.ieee.org/document/9854067">https://ieeexplore.ieee.org/document/9854067</a>


Please Cite it as following

S. Wazir and M. M. Fraz, "HistoSeg: Quick attention with multi-loss function for multi-structure segmentation in digital histology images," 2022 12th International Conference on Pattern Recognition Systems (ICPRS), 2022, pp. 1-7, doi: 10.1109/ICPRS54038.2022.9854067.

## Histological Image Segmentation
This repo contains the code to Test and Train the HistoSeg <br>

HistoSeg is an Encoder-Decoder DCNN which utilizes the novel Quick Attention Modules and Multi Loss function to generate segmentation masks from histopathological images with greater accuracy.

## HistoSeg Qualitative Results

<img align="center" src="HistoSeg_Results.png" title="HistoSeg Qualitative Results">

## HistoSeg Learning Curve

<p align="left">
  <img src="HistoSeg_Loss.jpg" width="500" title="HistoSeg Learning Curve">
</p>

## HistoSeg Quantitative Results

<table>
<thead>
  <tr>
    <th colspan="3">MoNuSeg</th>
    <th colspan="3">GlaS</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>F1</td>
    <td>IoU</td>
    <td>Dice</td>
    <td>F1</td>
    <td>IoU</td>
    <td>Dice</td>
  </tr>
  <tr>
    <td>75.08</td>
    <td>71.06</td>
    <td>95.20</td>
    <td>98.07</td>
    <td>76.73</td>
    <td>99.09</td>
  </tr>
</tbody>
</table>





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
##
<a href="https://github.com/saadwazir/HistoSeg"><img src="https://img.shields.io/badge/HistoSeg-Quick_attention_with_multi--loss_function_for_multi--structure_segmentation_in_digital_histology_images-FFC300" alt="HistoSeg - Quick attention with multi-loss function for multi-structure segmentation in digital histology images"></a>


