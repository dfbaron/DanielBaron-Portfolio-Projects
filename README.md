
# WGAN-ResNet152

Implementation of Low Dose CT Image Denoising using a Generative Adversarial Network with Wasserstein Distance and Perceptual Loss. We use ResNet152 as pretrained Network to feature extraction for Perceptual Loss calculation.

## Dataset
The dataset need to be stored in the following way:

```bash
dataset
├── L067
│   ├── quarter_3mm
│   │       ├── L067_QD_3_1.CT.0004.0001 ~ .IMA
│   │       ├── L067_QD_3_1.CT.0004.0002 ~ .IMA
│   │       └── ...
│   └── full_3mm
│           ├── L067_FD_3_1.CT.0004.0001 ~ .IMA
│           ├── L067_FD_3_1.CT.0004.0002 ~ .IMA
│           └── ...
├── L096
│   ├── quarter_3mm
│   │       ├── L096_QD_3_1.CT.0004.0001 ~ .IMA
│   │       ├── L096_QD_3_1.CT.0004.0002 ~ .IMA
│   │       └── ...
│   └── full_3mm
│           ├── L096_FD_3_1.CT.0004.0001 ~ .IMA
│           ├── L096_FD_3_1.CT.0004.0002 ~ .IMA
│           └── ...
...
│
└── L506
   ├── quarter_3mm
   │       ├── L506_QD_3_1.CT.0004.0001 ~ .IMA
   │       ├── L506_QD_3_1.CT.0004.0002 ~ .IMA
   │       └── ...
   └── full_3mm
           ├── L506_FD_3_1.CT.0004.0001 ~ .IMA
           ├── L506_FD_3_1.CT.0004.0002 ~ .IMA   
           └── ...
```
Each folder corresponds to images of Full-Dose and Quarter-Dose for each patient. The dataset is available in the following [link](https://drive.google.com/file/d/1BeoP_BiVXBEb6_DAWOgbHbCddMjzsN_C/view?usp=sharing).

## Setup
To convert dicom files to numpy arrays, run the following script.

```bash
python prep.py
```

In addition, the models trained are available in the next [link](https://drive.google.com/drive/folders/1eeWaEjo-qXBeytGw3QduoYunwGgn1n47?usp=sharing).

## Execution
For testing and demo execution, we just take into account the two best models generated in the experimental phase. Those models are:


| Backbone Perceptual Loss | Patch Size | Number of patches | Discriminator  |
|--------------------------|------------|-------------------|----------------|
| ResNet-152               | (120, 120) | 8                 | Enabled        |
| VGG-19                   | (120, 120) | 8                 | Enabled        |

### Testing
```bash
python main.py --mode=test --backbone=vgg19
```

### Demo
```bash
python main.py --mode=demo --backbone=vgg19 --img=1
```
