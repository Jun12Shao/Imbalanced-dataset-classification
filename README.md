# Imbalanced-dataset-classification
In this project, we designed a Fully convolutional Network (FCN) and a ResNet+FCN for imbalanced dataset classification. 
Resampling technologies: Synthetic Minority Over-sampling Technique (SMOTE) and Adaptive Synthetic Sampling (ADASYN) as well as Cost-sensitive Loss were adopted to overcome the problem of imbalanced dataset.


## Training a Fully Convolutional Network

### Install dependencies

* Install [PyTorch and torchvision](http://pytorch.org/).
* Install imblearn, PIL, opencv-python,onnx, onnxruntime


### Set up the wood knots dataset
### Train models

(1) basic FCN
* Run `python FCNs.py --oversampling=False --cost_sensitive=False --best_model='cnn-1'`. This will run training, place the results into `results` and save the best model in "checkpoints". .

(2) basic FCN+cost-sensitive loss
* Run `python FCNs.py --oversampling=False --cost_sensitive=True --best_model='cnn-1'`. This will run training, place the results into `results` and save the best model in "checkpoints". .

(3) basic FCN+SMOTE
* Run `python FCNs.py --oversampling=True --cost_sensitive=False --oversampling_mode='SMOTE' --best_model='cnn-1'`. This will run training, place the results into `results` and save the best model in "checkpoints". .

(4) basic FCN+ADASYN
* Run `python FCNs.py --oversampling=True --cost_sensitive=False --oversampling_mode='ADASYN' --best_model='cnn-1'`. This will run training, place the results into `results` and save the best model in "checkpoints". .

(5) basic ResNet+FCN+ADASYN
* Run `python RestNet-FCNs.py --oversampling=True --cost_sensitive=False --oversampling_mode='ADASYN' --best_model='res-ADASYB-1'`. This will run training, place the results into `results` and save the best model in "checkpoints". .


### Evaluate
(1) For FCNs
* Run evaluation as: `python test_FCNs.py --best_model= './checkpoints/cnn-os-SMOTE-2.onnx' `.

(2) For ResNet+FCNs
* Run evaluation as: `python test_Res_FCNs.py --best_model= './checkpoints/res-ADASYN-3.onnx' `.
