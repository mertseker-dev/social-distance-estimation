# social-distance-estimation
This repository contains code and tutorial for automatic social distance estimation from single RGB images by using YOLOv4 object detector and OpenPose human pose estimator. 
## Contents
1. [Getting Started](#getting-started)
1. [Installation](#installation)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Usage](#quick-usage)
5. [Citation](#citation)

## Getting Started
The code requires the following libraries to be installed:

-  python 3.8
-  tensorflow 2.3.1
-  opencv 4.4.0.44
-  numpy 1.18.5

The code requires YOLOv4 and OpenPose models to be installed. Refer to https://github.com/AlexeyAB/darknet and https://github.com/CMU-Perceptual-Computing-Lab/openpose for installation instructions. After installations, download the 3 scripts automatic_evaluation_API.py, evaluate_labeled_images.py and evaluate_unlabeled_images.py from this page. Finally, the project folder should look like:

```sh
${project_dir}/
├── labels
│   ├── body_pixel_locations.csv
│   ├── camera_locations_photoshoot_identifiers.csv
│   ├── ground_truth_locations.csv
├── openpose_models
│   └── cameraParameters
│       └── flir
│            ├── 17012332.xml.example
│   └── face
│       ├── haarcascade_frontalface_alt.xml
│       ├── pose_deploy.prototxt
│       ├── pose_iter_116000.caffemodel
│   └── hand
│       ├── pose_deploy.prototxt
│       ├── pose_iter_102000.caffemodel
│   └── pose
│       └── body_25
│           ├── pose_deploy.prototxt
│           ├── pose_iter_584000.caffemodel
│       └── coco
│           ├── pose_deploy_linevec.prototxt
│       └── mpi
│           ├── pose_deploy_linevec.prototxt
│           ├── pose_deploy_linevec_faster_4_stages.prototxt
│   ├── getModels.bat
│   ├── getModels.sh
├── outputs_labeled_data
├── outputs_unlabeled_data
├── yolo_models
│   ├── coco.names.txt
│   ├── yolov4.cfg.txt
│   ├── yolov4.weights
├── automatic_evaluation_API.py
├── boost_filesystem-vc141-mt-gd-x64-1_69.dll
├── boost_filesystem-vc141-mt-x64-1_69.dll
├── boost_thread-vc141-mt-gd-x64-1_69.dll
├── boost_thread-vc141-mt-x64-1_69.dll
├── caffe.dll
├── caffe-d.dll
├── caffehdf5.dll
├── caffehdf5_D.dll
├── caffehdf5_hl.dll
├── caffehdf5_hl_D.dll
├── caffezlib1.dll
├── caffezlibd1.dll
├── cublas64_100.dll
├── cudart64_100.dll
├── cudnn64_7.dll
├── curand64_100.dll
├── evaluate_labeled_images.py
├── evaluate_unlabeled_images.py
├── gflags.dll
├── gflagsd.dll
├── glog.dll
├── glogd.dll
├── libgcc_s_seh-1.dll
├── libgfortran-3.dll
├── libopenblas.dll
├── libquadmath-0.dll
├── opencv_videoio_ffmpeg420_64.dll
├── opencv_world420.dll
├── opencv_world420d.dll
├── openpose.dll
├── openpose_python.py
├── pyopenpose.cp38-win_amd64.pyd
├── pyopenpose.exp
├── pyopenpose.lib
├── VCRUNTIME140.dll
```
