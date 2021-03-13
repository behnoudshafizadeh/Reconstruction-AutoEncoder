# Reconstruction-AutoEncoder
using autoencoder for iranian character reconstruction

## DISCRIPTION
> in this project,we use autoencoder for reconstruction task,specially for improving distorted characters in license-plate lead to improve character recognition in object detection(YOLOV3).autoencoder is divided two parts: 1) encoder 2)decoder,when the input image is forwarded to encode part,the size of image is decreased by convolution layer and max pooling layer and the compressed image is represented as latent vector,in contrast,the decode layeres have deconvolution and upsampling layers that will increase the size image,untill the size of input and output will equal.

## DATASET
> dataset is croped by me and my co-worker by using iranian license-plate dataset in different situations (illumination,angle,weather condition,...) in the left side of autoencoder ,we use distorted character,but in other side(output) we used correct image related to distorted characters,so the autoencoder will learn to decrease difference between input and output image by using MSE(mean square loss),in result,we used iranian characters in size 28*28 as gray-scale image,and we put in two different directrot as `clean_data` and `noisy_data`.in below we see the examples of dataset:

|              | dataset | 
| -----------  | -------- | 
| clean_data   | ![clean_data](https://user-images.githubusercontent.com/53394692/111038670-8723ab80-843f-11eb-98dc-c8dfc762a406.PNG) | 
| noisy_data   |  ![noisy_data](https://user-images.githubusercontent.com/53394692/111038687-9c98d580-843f-11eb-80f6-f3d519483db2.PNG) | 

## train and test procedure
> * first run `train_conv_autoencoder.py` by running below command in your terminal:
 ```
python train_conv_autoencoder.py
 ``` 
> * see the results of train/validation accuracy in below chart:
![result](https://user-images.githubusercontent.com/53394692/111039583-0ca95a80-8444-11eb-873e-9607e24b86f3.png)
> * after ending training,the weights file with `.model` saved in your directory,for testing procedure.
> * for testing procedure,set iamges in `test` directory and change directory path `path1` basis on your directory , use the `pred-autoencoder.ipynb` and `.model` weight file and run it cell by cell,basis on jupyter notebook.
> * after testing ,see the reconstruction result as below:
> * ![test](https://user-images.githubusercontent.com/53394692/111040017-614dd500-8446-11eb-8d9d-883ee4ba9aa8.PNG)
