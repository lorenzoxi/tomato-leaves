# 🍅🍃 Tomato leaves: a Vision and Cognitive Systems Project | a.a.2023-204

*In this work, we explore the classification of tomato leaf diseases using deep learning techniques to identify whether leaves are healthy or diseased. Utilizing a custom dataset from PlantDoc and PlantVillage repositories, we implement and evaluate various convolutional neural network (CNN) architectures and transformers-based models. We also implement the LSGNet model, a lightweight neural network optimized for computational efficiency and real-time application. Our experiments involve feature extraction, fine-tuning, and deployment on a Raspberry Pi to assess the models’ practical performance. This project highlights the potential of integrating advanced AI techniques in agriculture, especially for early disease detection and crop health management. Furthermore, the deployment of these models on an IoT devices underscores their practical utility in enhancing farm operations and promoting sustainable farming practices.*

[Read more about the project here](https://unipdit-my.sharepoint.com/:b:/g/personal/lorenzo_perinello_studenti_unipd_it/ERC7H9QT5GdIjAQ3e5nWoEQBvSo6oyrhcRRWPIUjiErZHQ?e=1MvZlz).

## Datasets

Datasets used for fine-tuning and features extraction:

- PlantVillage
- PlantDoc

Find more information about the datasets used
and download the whole dataset from [huggingface.co/tomato-leaves-dataset](https://huggingface.co/datasets/lorenzoxi/tomato-leaves-dataset) or
direclty using the following prompt:

```bash
curl -X GET \
     "https://datasets-server.huggingface.co/first-rows?dataset=lorenzoxi%2Ftomato-leaves-dataset&config=default&split=train"
```

## CNNs and ViTs

The following models were used for the experiments:

- ViT-Tiny-Patch16-224
- VGG16
- Swin-Tiny-Patch4-Window7-224
- ShuffleNetV2-x1.0
- ResNet50
- MobileViT-Small
- MobileNetV3-Large-100
- EfficientNetV2-RW-S
- AlexNet
- LSGNet

## Models download

You can find and download already-trained models directly from here: [download now](https://unipdit-my.sharepoint.com/:u:/g/personal/lorenzo_perinello_studenti_unipd_it/EYmm9epGQr5JsgH_CdouXD8BI-aNxM8JyyqaT6ZvG2gXMg?e=NcAT8b).
