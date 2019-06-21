## Exploring Character Shapes for Unsupervised Reconstruction of Strip-Shredded Text Documents

[Thiago M. Paixão](https://sites.google.com/site/professorpx), [Maria C. S. Boeres](http://www.inf.ufes.br/~boeres), [Cinthia O. A. Freitas](http://lattes.cnpq.br/1058846722790485), and [Thiago Oliveira-Santos](https://www.inf.ufes.br/~todsantos/home)

<!---Published in *todo*: [DOI](https://www.google.com/)-->
Paper published in IEEE Trans. on Information Forensics and Security. The manuscript is available [here](https://ieeexplore.ieee.org/document/8565900).

### BibTeX
<pre>
@article{paixao2018exploring,
  title={Exploring Character Shapes for Unsupervised Reconstruction of Strip-Shredded Text Documents},
  author={Paix{\~a}o, Thiago M and Boeres, Maria CS and Freitas, Cinthia OA and Oliveira-Santos, Thiago},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={14},
  number={7},
  pages={1744--1754},
  year={2018},
  publisher={IEEE}
}
</pre>

#### Abstract

Digital reconstruction of mechanically shredded documents has received increasing attention in the last years mainly for historical and forensics needs. Computational methods to solve this problem are highly desirable in order to mitigate the time-consuming human effort and to preserve document integrity. The reconstruction of strips-shredded documents is accomplished by horizontally splicing pieces so that the arising sequence (solution) is as similar as the original document. In this context, a central issue is the quantification of the fitting between the pieces (strips), which generally involves stating a function that associates a pair of strips to a real value indicating the fitting quality. This problem is also more challenging for text documents, such as business letters or legal documents, since they depict poor color information. The system proposed here addresses this issue by exploring character shapes as visual features for compatibility computation. Experiments conducted with real mechanically shredded documents showed that our approach outperformed in accuracy other popular techniques in the literature considering documents with (almost) only textual content.

---

### Reproducing the experiments

Although the system has several dependencies, the experiments can be easily reproduced thanks to the [Docker](https://www.docker.com/) container technology. After installing Docker in our environment, make sure you are able to run Docker containers as non-root user (check this [guide](https://docs.docker.com/install/linux/linux-postinstall) for additional information).







```
```


#### CycleGAN

The source code used for the CycleGAN model was made publicly available by [Van Huy](https://github.com/vanhuyz/CycleGAN-TensorFlow).

#### Faster R-CNN

The source code used for the Faster R-CNN model was made publicly available by [Xinlei Chen](https://github.com/endernewton/tf-faster-rcnn).

For training the Faster R-CNN, a pre-trained resnet-101 model was used to initializate the process an can be downloaded [here](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz).

---

### Trained Models

#### CycleGAN

The trained model used in this paper is available [here](https://drive.google.com/drive/folders/17CJ5-cOK2CteZTPtRaT7rfW8oSt38CCe?usp=sharing).

#### Faster R-CNN

The trained models used in this paper are available [here](https://drive.google.com/drive/folders/1XRtExg-QGVA-DFJ1EKf8L0GLCxe5wIqH?usp=sharing).

---

### Dataset

#### Berkeley Deep Drive Dataset

##### Dataset Acquisition

Download the Berkeley Deep Drive dataset [here](https://bdd-data.berkeley.edu/).
It is only necessary to download the Images and Labels files.

##### Dataset Filtering

After downloading the BDD dataset, the Images and Labels will be placed into the zipped files `bdd100k_images.zip` and `bdd100k_labels.zip` respectively. In the same directory, place the provided source code `filter_dataset.py` from this repository with the folder `lists`.

On the terminal, run: `python filter_dataset.py`.
It will take a few minutes, and at the end, the folder `images` and `labels` will contain the images and bounding boxes of the images respectively. 

#### Generated (Fake) Dataset

Available [here](https://drive.google.com/drive/folders/1ZoXfgpTT1N5eOsI4-Tcv0id3mqij5gsP?usp=sharing).

---

### Videos

Videos demonstrating the inference performed by the trained Faster R-CNN model which yielded the best results with our proposed system.

 Testing on Day+Night Dataset | Testing on Night Dataset 
:-------------------------:|:-------------------------:
[![Video1](https://github.com/viniciusarruda/cross-domain-car-detection/blob/master/images/day_plus_night_video_overview.png)](https://youtu.be/qENxVuUXa0s) Inferences performed on day+night dataset |  [![Video2](https://github.com/viniciusarruda/cross-domain-car-detection/blob/master/images/night_video_overview.png)](https://youtu.be/MqZ2I-h_FOA) Inferences performed on night dataset 


---

<!--### BibTeX-->

<!--Coming Soon !-->


<!--
    @article{berriel2017grsl,
        Author  = {Rodrigo F. Berriel and Andre T. Lopes and Alberto F. de Souza and Thiago Oliveira-Santos},
        Title   = {{Deep Learning Based Large-Scale Automatic Satellite Crosswalk Classification}},
        Journal = {IEEE Geoscience and Remote Sensing Letters},
        Year    = {2017},
        DOI     = {10.1109/LGRS.2017.2719863},
        ISSN    = {1545-598X},
    }
-->
