# social-distance-estimation
This repository contains code and tutorial for automatic social distance estimation from single RGB images by using YOLOv4 object detector and OpenPose human pose estimator. 
## Contents
1. [Getting Started](#getting-started)
2. [Dataset & Annotations](#dataset)
3. [Distance Evaluation on Annotated Data](#annotated)
4. [Distance Evaluation on Unannotated Data](#unannotated)
5. [Installation](#installation)
6. [Quick Usage](#quick-usage)
7. [Citation](#citation)

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

## Dataset & Annotations

- **Dataset**: 
We provide an annotated image dataset for testing purposes that can be used to evaluate our method and also any other social distance estimation method that can either output 3D location estimations for the people or the distance between the people. The images and the annotations can be downloaded from here: LINK. 

- **Annotations**:
The annotations are found in 3 seperate .csv files: body_pixel_locations.csv, camera_locations_photoshoot_identifiers.csv and ground_truth_locations.csv. Please download these .csv files from the link above and put all of them under a directory named 'labels' under your main project directory. Refer to the project directory schema above for clarity.  



## Distance Evaluation on Annotated Data

In order to run the code on annotated data (images with annotations), run the following command:

### Usage

```sh
$ python evaluate_labeled_images.py -r=[PATH_TO_IMAGES, SENSOR_WIDTH, SENSOR_HEIGHT, SCALE_PERCENT, SAFE_DISTANCE, FOCAL_LENGTH]
```
- PATH_TO_IMAGES (mandatory): Path to the folder that contains the images to be evaluated.
- SENSOR_WIDTH (mandatory): Sensor width of the camera(s) that were used to capture the images in PATH_TO_IMAGES, should be in mm units
- SENSOR_HEIGHT (mandatory): Sensor height of the camera(s) that were used to capture the images in PATH_TO_IMAGES, should be in mm units
- SCALE_PERCENT (mandatory): Integer between 0-100. 100 for no scaling, 50 for 50% downscaling etc. of the images before evaluation
- SAFE_DISTANCE (mandatory): Safe distance threshold for pair-wise distances. Any estimated pair-wise distance that is under this threshold will be regarded as a violation and it will be reported under the results. Should be in mm units
- FOCAL_LENGTH (optional): Focal length of the camera(s) that were used to capture the images in PATH_TO_IMAGES, should be in mm units

If you wish to run the code on the dataset we provide, please enter *36* for SENSOR_WIDTH and *24* for SENSOR_HEIGHT and do not enter any value for FOCAL_LENGTH.

### Output format
The outputs of the evaluation will automatically be written in a csv. file with a unique timestamp name under the folder "outputs_labeled_data". The naming format of the files are: "YYYYMMDD-HHMMSS.csv". The output file contains the following information for each image: filename, number of detected people, every estimated pair-wise distance, average estimated pair-wise distance, average pair-wise percentual distance estimation error, number of violations, person detection rate and false discovery rate for the people. All of the units are in mm.


