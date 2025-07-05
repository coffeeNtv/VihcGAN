# Virtual immunohistochemistry by conditional generative adversarial networks

This is the code base for "Virtual immunohistochemistry by conditional generative adversarial networks".



## Packages

dominate==2.6.0
h5py==3.7.0
numpy==1.21.5
opencv_python==4.6.0.66
Pillow==9.5.0
torch==1.11.0
torchvision==0.12.0
visdom==0.2.3

image-similarity-measures

piq

## Dataset set up

Put the paired images into the below format, where A stands for your source images and B stands for your target images. 

```html
├─Dataset
│  ├─trainA
│  ├─trainB
│  ├─testA
│  └─testB
```

## Training

python train.py --dataroot your_data_path --name your_model_name --niter 100 --niter_decay 200  --save_epoch_freq 50



## Evaluation

python test.py --dataroot your_data_path --name your_model_name --phase test --how_many testing_size --serial_batches --which_epoch latest

