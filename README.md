# awesome-3d-object-detection [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repository will share the classification, paper, code, and notes of awesome 3D object detection papers for everyone interested in 3D object detection. 

### keywords

- datasets

  - `KITTI`: KITTI; `NuScenes`: NuScenes; `Waymo`: Waymo; `ONCE`: ONCE; `Lyft`: Lyft; 

  - `Scan.`: ScanNet; `SUN.`: SUN RGB-D   
- type

  - `one`: one-stage; `two`: two-stage; 
  - `point`: point-based; `voxel`: voxel-based; `pv`: fusion-based
  - `L`: LiDAR; `I`: Image; `M`: Multi-sensor
  - `self`: self-attention; `tran.`: transform
- other
  - `CVPRW`: CVPR Workshop
  - `tf-code`: Tensorflow code; `pt-code`: PyTorch code


## Overview

- [2017](#2017)
- [2018](#2018)
- [2019](#2019)
- [2020](#2020)
- [2021](#2021)
- [2022](#2022)

## 2017

- **CVPR** **`[PointNet]`** Deep Learning on Point Sets for 3D Classification and Segmentation
- **NeurIPS** **`[PointNet++]`** Deep Hierarchical Feature Learning on Point Sets in a Metric Space
- **CVPR** 3D Bounding Box Estimation Using Deep Learning and Geometry
- **CVPR** Multi-View 3D Object Detection Network for Autonomous Driving
- **CVPR** `[OctNet]` Learning Deep 3D Representations at High Resolutions
- **ICCV** 2D-Driven 3D Object Detection in RGB-D Images
- **ICCV** **`[SSD-6D]`** Making RGB-Based 3D Detection and 6D Pose Estimation Great Again

## 2018

- **CVPR** Multi-Level Fusion based 3D Object Detection from Monocular Images
- **CVPR** PIXOR Real-time 3D Object Detection from Point Clouds
- **CVPR** Real-Time Seamless Single Shot 6D Object Pose Prediction
- **CVPR** **`[Frustum PointNets]`** Frustum PointNets for 3D Object Detection from RGB-D Data
- **CVPR** **`[VoxelNet]`** End-to-End Learning for Point Cloud Based 3D Object Detection
- **ECCV** Deep Continuous Fusion for Multi-Sensor 3D Object Detection
- **arXiv** `[AVOD]` Joint 3D Proposal Generation and Object Detection from View Aggregation
- **arXiv** Orthographic Feature Transform for Monocular 3D Object Detection
- **Sensors** **`[SECOND]`** Sparsely Embedded Convolutional Detection
- **arXiv** `[SqueezeSeg]` Convolutional Neural Nets with Recurrent CRF for Real-Time Road-Object Segmentation from 3D LiDAR Point Cloud
- **IEEE ITSC** Towards Safe Autonomous Driving Capture Uncertainty in the Deep Neural Network For Lidar 3D Vehicle Detection

## 2019

- **arXiv** A Survey on 3D Object Detection Methods for Autonomous Driving Applications
- **arXiv** **`[CBGS]`** Class-balanced Grouping and Sampling for Point Cloud 3D Object Detection
- **arXiv** **`[Complexer-YOLO]`** Real-Time 3D Object Detection and Tracking on Semantic Point Clouds
- **arXiv** **`[Part-A^2 Net]`** 3D Part-Aware and Aggregation Neural Network for Object Detection from Point Cloud
- **CVPR** 3D Point Capsule Networks
- **CVPR** Bounding Box Regression with Uncertainty for Accurate Object Detection
- **CVPR** DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
- **CVPR** PointFusion Deep Sensor Fusion for 3D Bounding Box Estimation
- **CVPR** ROI-10D Monocular Lifting of 2D Detection to 6D Pose and Metric Shape
- **CVPR** Stereo R-CNN based 3D Object Detection for Autonomous Driving
- **CVPR** **`[Frustum ConvNet]`** Sliding Frustums to Aggregate Local Point-Wise Features for Amodal 3D Object Detection
- **CVPR** **`[GSPN]`** Generative Shape Proposal Network for 3D Instance Segmentation in Point Cloud
- **CVPR** **`[LaserNet]`** An Efficient Probabilistic 3D Object Detector for Autonomous Driving
- **CVPR** **`[PlaneRCNN]`** 3D Plane Detection and Reconstruction from a Single Image
- **CVPR** **`[PointConv]`** Deep Convolutional Networks on 3D Point Clouds
- **CVPR** **`[PointPillars]`** Fast Encoders for Object Detection from Point Clouds
- **CVPR** **`[PointRCNN]`** 3D Object Proposal Generation and Detection From Point Cloud
- **CVPR** **`[PointWeb]`** Enhancing Local Neighborhood Features for Point Cloud Processing
- **CVPR** **`[RoarNet]`** A Robust 3D Object Detection based on RegiOn Approximation Refinement
- **CVPR** **`[UberATG-MMF]`** Multi-Task Multi-Sensor Fusion for 3D Object Detection
- **ICCV** `[DPOD]` 6D Pose Object Detector and Refiner
- **IEEE RA-L** Focal Loss in 3D Object Detection
- **ICCV** (oral) **`[ShellNet]`** Efficient Point Cloud Convolutional Neural Networks using Concentric Shells Statistics
- **ICCV** Joint Monocular 3D Detection and Tracking
- **ICCV** **`[KPConv]`** Flexible and Deformable Convolution for Point Clouds
- **ICCV** **`[STD]`** Sparse to Dense 3D Object Detector for Point Cloud
- **ICCV** **`[VoteNet]`** Deep Hough Voting for 3D Object Detection in Point Clouds
- **ICRA** **`[MVX-Net]`** Multimodal VoxelNet for 3D Object Detection
- **IROS** **`[Frustum ConvNet]`** Sliding Frustums to Aggregate Local Point-Wise Features for Amodal 3D Object Detection
- **NeurIPS** **`[3D-BoNet]`** Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds
- **NeurIPS** **`[PVCNN]`** Point-Voxel CNN for Efficient 3D Deep Learning
- **Neurocomputing** **`[SARPNET]`** Shape Attention Regional Proposal Network for 0LiDAR-based 3D Object 0Detection
- **PAMI** Deep Learning for 3D Point Clouds- A Survey
- **arXiv** **`[3D IoU Loss]`** IoU Loss for 2D or 3D Object Detection
- **arXiv** **`[IPOD]`** An Industrial and Professional Occupations Dataset and its Applications to Occupational Data Mining and Analysis
- **arXiv** **`[SqueezeSegV2]`** Improved Model Structure and Unsupervised Domain Adaptation for Road-Object Segmentation from a LiDAR Point Cloud

## 2020

- **3DV** **`[PanoNet3D]`** Combining Semantic and Geometric Understanding for LiDARPoint Cloud Detection
- **3DV** **`[SF-UDA3D]`**  Source-Free Unsupervised Domain Adaptation for LiDAR-Based 3D Object Detection
- **AAAI** (Oral) **`[TANet]`** Robust 3D Object Detection from Point Clouds with Triple Attention
- **AAAI** **`[PI-RCNN]`** An Efficient Multi-sensor 3D Object Detector with Point-based Attentive Cont-conv Fusion Module
- **ACM MM** Weakly Supervised 3D Object Detection from Point Clouds
- **arXiv** An Overview Of 3D Object Detection
- **arXiv** Boundary-Aware Dense Feature Indicator for Single-Stage 3D Object Detection from Point Clouds
- **arXiv** Part-Aware Data Augmentation for 3D Object Detection in Point Cloud
- **arXiv** Quantifying Data Augmentation for LiDAR based 3D Object Detection
- **arXiv** **`[3D IoU-Net]`** IoU Guided 3D Object Detector for Point Clouds
- **arXiv** **`[3D-CVF]`** Generating Joint Camera and LiDAR Features Using Cross-View Spatial Feature Fusion for 3D Object Detection
- **arXiv** **`[Associate-3Ddet]`** Perceptual-to-Conceptual Association for 3D Point Cloud Object Detection
- **arXiv** **`[CenterNet3D]`** An Anchor free Object Detector for Autonomous Driving
- **arXiv** **`[CenterPoint]`** Center-based 3D Object Detection and Tracking
- **arXiv** **`[DPointNet]`** A Density-Oriented PointNet for 3D Object Detection in Point Clouds
- **arXiv** **`[Finding Your (3D) Center]`** 3D Object Detection Using a Learned Loss
- **arXiv** **`[MoCa]`** Multi-Modality Cut and Paste for 3D Object Detection
- **arXiv** **`[MVAF-Net]`** Multi-View Adaptive Fusion Network for 3D Object Detection
- **arXiv** **`[Object as Hotspots]`** An Anchor-Free 3D Object Detection Approach via Firing of Hotspots
- **arXiv** **`[PCT]`** Point Cloud Transforme
- **arXiv** **`[RangeRCNN]`** Towards Fast and Accurate 3D Object Detection with Range Image Representation
- **arXiv** **`[SVGA-Net]`** Sparse Voxel-Graph Attention Network for 3D Object Detection from Point Clouds
- **CVPR** (Oral) **`[3DSSD]`** Point-based 3D Single Stage Object Detector
- **CVPR** (Oral) **`[PointAugment]`** an Auto-Augmentation Framework for Point Cloud Classification
- **CVPR** (Oral) **`[PointGroup]`** Dual-Set Point Grouping for 3D Instance Segmentation
- **CVPR** (Oral) **`[What You See is What You Get]`** Exploiting Visibility for 3D Object Detection
- **CVPR** End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection
- **CVPR** Learning multiview 3D point cloud registration
- **CVPR** **`[3DVID]`** LiDAR-based Online 3D Video Object Detection with Graph-based Message Passing and Spatiotemporal Transformer Attention
- **CVPR** **`[DSGN]`** Deep Stereo Geometry Network for 3D Object Detection
- **CVPR** **`[HVNet]`** Hybrid Voxel Network for LiDAR Based 3D Object Detection
- **CVPR** **`[ImVoteNet]`** Boosting 3D Object Detection in Point Clouds With Image Votes
- **CVPR** **`[PF-Net]`** Point Fractal Network for 3D Point Cloud Completion
- **CVPR** **`[Point-GNN]`** Graph Neural Network for 3D Object Detection in a Point Cloud
- **CVPR** **`[PointPainting]`** Sequential Fusion for 3D Object Detection
- **CVPR** **`[PolarNet]`** An Improved Grid Representation for Online LiDAR Point Clouds Semantic Segmentation
- **CVPR** **`[PV-RCNN]`** Point-Voxel Feature Set Abstraction for 3D Object Detection
- **CVPR** **`[PVN3D]`** A Deep Point-wise 3D Keypoints Voting Network for 6DoF Pose Estimation
- **CVPR** **`[RandLA-Net]`** Efficient Semantic Segmentation of Large-Scale Point Clouds
- **CVPR** **`[SA-SSD]`** Structure Aware Single-stage 3D Object Detection from Point Cloud
- **CVPR** **`[SERCNN]`** Joint 3D Instance Segmentation and Object Detection for Autonomous Driving
- **CVPRW** **`[AFDet]`** Anchor Free One Stage 3D Object Detection
- **CVPRW** **`[PV-RCNN]`** The Top-Performing LiDAR-only Solutions for 3D Detection & 3D Tracking & Domain Adaptation of Waymo Open Dataset Challenges
- **ECCV** Active Perception using Light Curtains for Autonomous Driving
- **ECCV** An LSTM Approach to Temporal 3D Object Detection in LiDAR Point Clouds
- **ECCV** Reinforced Axial Refinement Network for Monocular 3D Object Detection
- **ECCV** Rotation-robust Intersection over Union for 3D Object Detection
- **ECCV** Streaming Object Detection for 3-D Point Clouds
- **ECCV** **`[Deformable PV-RCNN]`** Improving 3D Object Detection with Learned Deformations
- **ECCV** **`[EPNet]`** Enhancing Point Features with Image Semantics for 3D Object Detection
- **ECCV** **`[InfoFocus]`** 3D Object Detection for Autonomous Driving with Dynamic Information Modeling
- **ECCV** **`[Object as Hotspots]`** An Anchor-Free 3D Object Detection Approach via Firing of Hotspots
- **ECCV** **`[POD]`** Pillar-based Object Detection for Autonomous Driving
- **ECCV** **`[PPBA]`** Improving 3D Object Detection through Progressive Population Based Augmentation
- **ECCV** **`[SSN]`** Shape Signature Networks for Multi-class Object Detection from Point Clouds
- **ECCV** **`[WS3D]`** Weakly Supervised 3D Object Detection from Lidar Point Cloud
- **IROS** **`[CLOCs]`** Camera-LiDAR Object Candidates Fusion for 3D Object Detection
- **IROS** **`[MVLidarNet]`** Real-Time Multi-Class Scene Understanding for Autonomous Driving Using Multiple Views
- **NeurIPS** Group Contextual Encoding for 3D Point Clouds
- **Neurocomputing** Multi-view semantic learning network for point cloud based 3D object detection
- **PAMI** **`[From Points to Parts]`** 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
- **Sensors** **`[Voxel-FPN]`** multi-scale voxel feature aggregation in 3D object detection from point clouds
- **SensorsCouncil** Fusion of 3D LIDAR and Camera Data for Object Detection in Autonomous Vehicle Applications

## 2021

-  **AAAI** Self-supervised Multi-view Stereo via Effective Co-segmentation and Data-Augmentation
-  **AAAI** **`[CIA-SSD]`** Confident IoU-Aware Single-Stage Object Detector From Point Cloud
-  **AAAI** **`[Voxel R-CNN]`** Towards High Performance Voxel-based 3D Object Detection
-  **ACM MM** **`[From Voxel to Point]`** IoU-guided 3D Object Detection for Point Cloud with Voxel-to-Point Decoder
-  **arXiv** Multi-Modal 3D Object Detection in Autonomous Driving a Survey
-  **arXiv** **`[AF2-S3Net]`** Attentive Feature Fusion with Adaptive Feature Selection for Sparse Semantic Segmentation Network
-  **arXiv** **`[BANet]`** Boundary-Aware 3D Object Detection from Point Clouds
-  **arXiv** **`[BEVDetNet]`** Bird's Eye View LiDAR Point Cloud based Real-time 3D Object
-  **arXiv** **`[CenterAtt]`** Fast 2-stage Center Attention Network A solution for Waymo Open Dataset Real-time 3D Detection Challenge
-  **arXiv** **`[M3DeTR]`** Multi-representation, Multi-scale, Mutual-relation 3D Object Detection with Transformers
-  **arXiv** **`[MGAF-3DSSD]`** Anchor-free 3D Single Stage Detector with Mask-Guided Attention for Point Cloud
-  **arXiv** **`[PV-RCNN++]`** Point-Voxel Feature Set Abstraction With Local Vector Representation for 3D Object Detection
-  **arXiv** **`[R-AGNO-RPN]`** A LIDAR-Camera Region Deep Network for Resolution-Agnostic Detection
-  **arXiv** **`[SA-Det3D]`** Self-Attention Based Context-Aware 3D Object Detection
-  **arXiv** **`[SARFE]`** Structure Information is the Key Self-Attention RoI Feature Extractor in 3D Object Detection
-  **arXiv** **`[SFUDA]`** Attentive Prototypes for Source-free Unsupervised Domain Adaptive 3D Object Detection
-  **arXiv** **`[SIENet]`** Spatial Information Enhancement Network for 3D Object Detection from Point Cloud
-  **arXiv** **`[SST]`** Embracing Single Stride 3D Object Detector with Sparse Transformer
-  **arXiv** **`[VPFNet]`** Improving 3D Object Detection with Virtual Point based LiDAR and Stereo Data Fusion
-  **arXiv** **`[VPFNet]`** Voxel-Pixel Fusion Network for Multi-class 3D Object Detection
-  **CVPR**  **`[PointAugmenting]`** Cross-Modal Augmentation for 3D Object Detection
-  **CVPR**  **`[PVGNet]`** A Bottom-Up One-Stage 3D Object Detector with Integrated Multi-Level Features
-  **CVPR (oral)** **`[CaDDN]`** Categorical Depth Distribution Network for Monocular 3D Object Detection
-  **CVPR** Back-tracing Representative Points for Voting-based 3D Object Detection in Point Clouds
-  **CVPR** Offboard 3D Object Detection from Point Cloud Sequences
-  **CVPR** **`[3DIoUMatch]`** Leveraging IoU Prediction for Semi-Supervised 3D Object Detection
-  **CVPR** **`[BRNet]`** Back-tracing Representative Points for Voting-based 3D Object Detection in Point Clouds
-  **CVPR** **`[Centerpoint]`** Center-based 3D Object Detection and Tracking
-  **CVPR** **`[EBM3DOD]`** Accurate 3D Object Detection using Energy-Based Models
-  **CVPR** **`[HVPR]`** Hybrid Voxel-Point Representation for Single-stage 3D Object Detection
-  **CVPR** **`[LiDAR R-CNN]`** An Efficient and Universal 3D Object Detector
-  **CVPR** **`[SE-SSD]`** Self-Ensembling Single-Stage Object Detector From Point Cloud
-  **CVPR** **`[ST3D]`** Self-training for Unsupervised Domain Adaptation on 3D ObjectDetection
-  **CVPRW** **`[Waymo 1st]`** 1st Place Solutions to the Real-time 3D Detection and the Most Efficient Model of the Waymo Open Dataset Challenges 2021
-  **CVPRW** **`[Waymo 2nd]`** **`[CenterPoint++]`** submission to the Waymo Real-time 3D Detection Challenge
-  **CVPRW** **`[Waymo 3rd]`** 3rd Place Solution of Waymo Open Dataset Challenge 2021Real-time 3D Detection Track
-  **ICCV** **`[4D-Net]`** 4D-Net for Learned Multi-Modal Alignment
-  **ICCV** **`[CT3D]`** Improving 3D Object Detection with Channel-wise Transformer
-  **ICCV** **`[Pyramid R-CNN]`** Towards Better Performance and Adaptability for 3D Object Detection
-  **ICCV** **`[SPG]`** Unsupervised Domain Adaptation for 3D Object Detection via Semantic Point Generation
-  **ICCV** **`[VoTr]`** Voxel Transformer for 3D Object Detection
-  **ICCV** **`[You Don’t Only Look Once]`** Constructing Spatial-Temporal Memory for Integrated 3D Object Detection and Tracking
-  **NeurIPS** **`[MVP]`** Multimodal Virtual Point 3D Detection
-  **IEEE TCSVT** **`[From Multi-View to Hollow-3D]`** Hallucinated Hollow-3D R-CNN for 3D Object Detection
-  **ICCV 2021** **`[Pyramid R-CNN]`**  Towards Better Performance and Adaptability for 3D Object Detection 
-  **arXiv 2021**  **`[M3DeTR]`**  Multi-representation, Multi-scale, Mutual-relation 3D Object Detection with Transformers
-  **arXiv 2021** **`[SST]`** Embracing Single Stride 3D Object Detector with Sparse Transformer

## 2022

- **AAAI** **`[BtcDet]`** Behind the Curtain Learning Occluded Shapes for 3D Object Detection
- **arXiv** **`[PiFeNet]`** Pillar-Feature Network for Real-Time 3D Pedestrian Detection from Point Cloud
- **AAAI** **`[SASA]`** Semantics-Augmented Set Abstraction for Point-based 3D Object Detection
- **WACV** **`[Fast-CLOCs]`** Fast Camera-LiDAR Object Candidates Fusion for 3D Object Detection
- **CVPR** A Versatile Multi-View Framework for LiDAR-based 3D Object Detection with Guidance from Panoptic Segmentation
- **CVPR** Pseudo-Stereo for Monocular 3D Object Detection in Autonomous Driving
- **CVPR** **`[DeepFusion]`** Lidar-Camera Deep Fusion for Multi-Modal 3D Object Detection
- **CVPR** **`[MonoJSG]`** Joint Semantic and Geometric Cost Volume for Monocular 3D Object Detection
- **CVPR** **`[PDV]`** Point Density-Aware Voxels for LiDAR 3D Object Detection
- **CVPR** **`[SST]`** Embracing Single Stride 3D Object Detector with Sparse Transformer
- **arXiv** Dense Voxel Fusion for 3D Object Detection
- **arXiv** **`[CG-SSD]`** Corner Guided Single Stage 3D Object Detection from LiDAR Point Cloud
- **arXiv** **`[SASA]`** Semantics-Augmented Set Abstraction for Point-based 3D Object Detection