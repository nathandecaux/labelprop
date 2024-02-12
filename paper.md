---
title: 'LabelProp: A semi-automatic segmentation tool for 3D medical images'
tags:
  - Python
  - segmentation
  - deep learning
  - medical images
  - musculoskeletal 

authors:
  - name: Nathan Decaux
    orcid: 0000-0002-6911-6373
    affiliation: "1, 2"
  - name: Pierre-Henri Conze
    orcid: 0000-0003-2214-3654
    affiliation: "1, 2"
  - name: Juliette Ropars
    orcid: 0000-0001-7467-759X
    affiliation: "1, 3"
  - name: Xinyan He
    affiliation: "2"
  - name: Frances T. Sheehan
    affiliation: "4"
  - name: Christelle Pons
    orcid: 0000-0003-3924-6035
    affiliation: "1, 3, 5"
  - name: Douraied Ben Salem
    orcid: 0000-0001-5532-2208
    affiliation: "1, 3"
  - name: Sylvain Brochard
    orcid: 0000-0002-4950-1696
    affiliation: "1, 3"
  - name: François Rousseau
    orcid: 0000-0001-9837-7487
    affiliation: "1, 2"

affiliations:
  - name: LaTIM UMR 1101, Inserm, Brest, France
    index: 1
  - name: IMT Atlantique, Brest, France
    index: 2
  - name: University Hospital of Brest, Brest, France
    index: 3
  - name: Rehabilitation Medicine, NIH, Bethesda, USA
    index: 4
  - name: Fondation ILDYS, Brest, France
    index: 5

date: 08 January 2024
bibliography: paper.bib


---

# Summary

LabelProp is a tool that provides a semi-automated method to segment 3D medical images with multiple labels. It is a convenient implementation of our peer-reviewed method designed to assist medical professionals in segmenting musculoskeletal structures on scans based on a small number of annotated slices [@decaux_semi-automatic_2023]. LabelProp leverages deep learning techniques, but can be used without a training set. It is available as a PyPi package and offers both a command-line interface (CLI) and an API. Additionally, LabelProp provides two plugins, namely 'napari-labelprop' and 'napari-labelprop-remote', which facilitate training and inference on a single scan within the multi-dimensional viewer Napari. It is available on GitHub with pretrained weights ([https://github.com/nathandecaux/napari-labelprop](https://github.com/nathandecaux/napari-labelprop))

# Statement of need

Segmenting musculoskeletal structures from MR images is crucial for clinical research, diagnosis, and treatment planning. However, challenges arise from the limited availability of annotated datasets, particularly in rare diseases or pediatric cohorts [@conze2020healthy]. While manual segmentation ensures accuracy, it is labor-intensive and prone to observer variability [@vadineanu_analysis_2022]. Existing semi-automatic methods based on point and scribbles require minimal interactions but often lack reproducibility [@sakinis2019interactive; @zhang2021interactive; @lee_scribble2label_2020 ;@chanti_ifss-net_2021].

LabelProp addresses these challenges with a novel deep registration-based label propagation method. This approach efficiently adapts to various musculoskeletal structures, leveraging image intensity and muscle shape for improved segmentation accuracy.

A key innovation of LabelProp is its minimal reliance on manual annotations. Demonstrating the capability for accurate 3D segmentation from as few as three annotated slices per MR volume [@decaux_semi-automatic_2023], it significantly reduces the workload for medical professionals and is particularly beneficial where extensive annotated data is scarce. This feature aligns with the method of slice-to-slice registration [@ogier2017individual], but is further enhanced by deep learning techniques.

Similar to Voxelmorph, the underlying model in this approach learns to generate deformations without supervision [@balakrishnan2019voxelmorph]. However, it specifically focuses on aligning adjacent 2D slices and can be trained directly on the scan that needs to be segmented or on a complete dataset for optimal results. When training the model with at least two annotations for a scan, a constraint is added to ensure that the generated deformations are consistent from both an image and segmentation perspective. Additionally, weak annotations in the form of scribbles can be provided during training on intermediate slices to provide additional guidance for propagation. Examples of manual annotations and scribbles are shown in Fig. 1. 

 During the inference phase, each annotation is propagated to its nearest neighboring annotation, resulting in two predictions for each intermediate slice from different source annotations. The label fusion process involves weighting each prediction based on their distance to the source annotation or an estimate of the registration accuracy. Importantly, the propagation method is label-agnostic, allowing for the simultaneous segmentation of multiple structures, regardless of whether they are manually annotated on the same slice or not.
 
 ![Example of propagation from 3 manual annotations of the deltoid muscle in a MRI, in axial plane. Optionnal scribbles (yellow) can be provided, without plane constraints, for further guidance.\label{fig:propagation}](propagation.pdf)

# State of the field
In a previous study, we evaluated our method against various approaches in a shoulder muscle MRI dataset and the publicly accessible MyoSegmenTUM dataset. Specifically, we focused on intra-subject segmentation using only 3 annotated slices [@decaux_semi-automatic_2023]. The reference methods were the [ITKMorphologicalContourInterpolation](https://github.com/KitwareMedical/ITKMorphologicalContourInterpolation) approach [@albu2008morphology], a well-known implementation of [UNet](https://github.com/milesial/Pytorch-UNet) [@ronneberger2015u], and a [semi-automatic image sequence segmentation approach](https://github.com/ajabri/videowalk) [@jabri_space-time_2020]. Our results showed that in this particular configuration, our method (Labelprop) outperformed all of these methods. Additionally, our method also demonstrated competitive performance compared to a leave-one-out trained UNet for the shoulder dataset [@conze2020healthy].


# Software Details

LabelProp is composed of three main components: labelprop, napari-labelprop, and napari-labelprop-remote. The labelprop algorithm is accompanied by a command-line interface (CLI) and a REST API. The CLI enables unsupervised pretraining or training with sparse annotations on a dataset, and inference on a single volume. The API provides access to training with annotations and inference on a single subject via HTTP requests. It is used in the napari-labelprop-remote plugin, but can be adapted to other extendable viewer/segmentation tools such as [3D Slicer](https://github.com/Slicer/Slicer) or [MITK](https://github.com/MITK/MITK). The napari-labelprop plugin brings the labelprop algorithm into the interactive Napari platform, allowing users to conduct both the training and inference stages of label propagation directly within the Napari environment. The napari-labelprop-remote plugin extends the functionality of napari-labelprop, allowing users to perform training and inference on a remote server through the labelprop API. These tools provide a versatile and user-friendly toolkit for 3D image segmentation, offering the flexibility to work locally or remotely, and leveraging deep learning to efficiently generate 3D delineations from slice annotations.


# Acknowledgements
This work was partially funded by ANR (AI4Child project, grant ANR-19-CHIA-0015) and Innoveo from CHRU de Brest.

# References