## Exploring Character Shapes for Unsupervised Reconstruction of Strip-Shredded Text Documents

[Thiago M. Paixão](https://sites.google.com/site/professorpx), [Maria C. S. Boeres](http://www.inf.ufes.br/~boeres), [Cinthia O. A. Freitas](http://lattes.cnpq.br/1058846722790485), and [Thiago Oliveira-Santos](https://www.inf.ufes.br/~todsantos/home)

<!---Published in *todo*: [DOI](https://www.google.com/)-->
Paper published in IEEE Trans. on Information Forensics and Security. The manuscript is available [here](https://ieeexplore.ieee.org/document/8565900).

### BibTeX
```
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
```

#### Abstract

Digital reconstruction of mechanically shredded documents has received increasing attention in the last years mainly for historical and forensics needs. Computational methods to solve this problem are highly desirable in order to mitigate the time-consuming human effort and to preserve document integrity. The reconstruction of strips-shredded documents is accomplished by horizontally splicing pieces so that the arising sequence (solution) is as similar as the original document. In this context, a central issue is the quantification of the fitting between the pieces (strips), which generally involves stating a function that associates a pair of strips to a real value indicating the fitting quality. This problem is also more challenging for text documents, such as business letters or legal documents, since they depict poor color information. The system proposed here addresses this issue by exploring character shapes as visual features for compatibility computation. Experiments conducted with real mechanically shredded documents showed that our approach outperformed in accuracy other popular techniques in the literature considering documents with (almost) only textual content.

---

### Reproducing the experiments

Although the system has several dependencies, the experiments can be easily reproduced thanks to the [Docker](https://www.docker.com/) container technology. After installing Docker in our environment, make sure you are able to run Docker containers as non-root user (check this [guide](https://docs.docker.com/install/linux/linux-postinstall) for additional information). Then, run the following bash commands in a terminal:

1. Clone the project repository and enter the project directory:
```
git clone https://github.com/thiagopx/docrec-tifs18.git
cd docrect-tifs18
```
2. Build the container (defined in ```docker/Dockerfile```):
```
bash build.sh
```
3. Run the experiments:
```
bash run.sh
```

*Technical note* : the threshold for shape matching is already calibrate acording the source code in ```train``` directory. The optimal value was obtained by running ```python train.py```, and the configuration file ```algorithms.cfg``` was manually modified accordingly.


---

### Sensitivity analysis

The sensitivity of some key parameters was analyzed in the paper using the one-factor-at-time (OFAT) approach. In this repository, we also attached the [file](https://github.com/thiagopx/docrec-tifs18/blob/master/sensitivity.pdf) with a more detailed analysis (including graphs) in response of one the reviewer's request.

