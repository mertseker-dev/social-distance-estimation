# social-distance-estimation
This repository contains code and tutorial for automatic social distance estimation from single RGB images by using YOLOv4 object detector and OpenPose human pose estimator. 
## Contents
1. [Getting Started](#getting-started)
2. [Dataset and Annotations](#dataset-and-annotations)
3. [Distance Evaluation on Annotated Data](#distance-evaluation-on-annotated-data)
4. [Distance Evaluation on Unannotated Data](#distance-evaluation-on-unannotated-data)
5. [Adding Your Own Annotated Images](#adding-your-own-annotated-images)
6. [Testing Your Own Method](#testing-your-own-method)
7. [Results](#results)
8. [References](#references)


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

## Dataset and Annotations

- **Dataset**: 
We provide an annotated image dataset for testing purposes that can be used to evaluate our method and also any other social distance estimation method that can output either 3D location estimations for the people or the distance between the people. The images and the annotations can be downloaded from here: LINK. 

- **Annotations**:
The annotations are found in 3 seperate .csv files: body_pixel_locations.csv, camera_locations_photoshoot_identifiers.csv and ground_truth_locations.csv. Please download these .csv files from the link above or from this page, note that they should be under a directory named 'labels' under your main project directory. Refer to the project directory schema above in [Getting Started](#getting-started) for clarity.  



## Distance Evaluation on Annotated Data

In order to run the code on annotated data (images with annotations), run the following command:

### Usage

```sh
$ python evaluate_labeled_images.py -r=[PATH_TO_IMAGES, SENSOR_WIDTH, SENSOR_HEIGHT, SCALE_PERCENT, SAFE_DISTANCE, FOCAL_LENGTH]
```
- PATH_TO_IMAGES (mandatory): Path to the folder that contains the images to be evaluated.
- SENSOR_WIDTH (mandatory): Sensor width of the camera(s) that were used to capture the images in PATH_TO_IMAGES, should be in mm units.
- SENSOR_HEIGHT (mandatory): Sensor height of the camera(s) that were used to capture the images in PATH_TO_IMAGES, should be in mm units.
- SCALE_PERCENT (mandatory): Integer between 0-100. 100 for no scaling, 50 for 50% downscaling etc. of the images before evaluation.
- SAFE_DISTANCE (mandatory): Safe distance threshold for pair-wise distances. Any estimated pair-wise distance that is under this threshold will be regarded as a violation and it will be reported under the results. Should be in mm units.
- FOCAL_LENGTH (optional): Focal length of the camera(s) that were used to capture the images in PATH_TO_IMAGES, should be in mm units.

If you wish to run the code on the dataset we provide, please enter *36* for SENSOR_WIDTH and *24* for SENSOR_HEIGHT and do not enter any value for FOCAL_LENGTH.

### Output format
The outputs of the evaluation will automatically be written in a csv. file with a unique timestamp name under the folder "outputs_labeled_data". The naming format of the files are: "YYYYMMDD-HHMMSS.csv". The output file contains the following information for each image: filename, number of detected people, every estimated pair-wise distance, average estimated pair-wise distance, average pair-wise percentual distance estimation error (obtained by comparing the estimated pair-wise distances with ground truth pair-wise distances), number of violations, person detection rate and false discovery rate for the people. Finally, the last row with filename 'ALL' contains the average of each column. All of the units are in mm.

## Distance Evaluation on Unannotated Data

You can run our method on any image as long as sensor dimensions and focal length information of the camera that was used to capture the image are known.
In order to run the code on unannotated data (images without annotations), run the following command:

### Usage

```sh
$ python evaluate_unlabeled_images.py -r=[PATH_TO_IMAGES, SENSOR_WIDTH, SENSOR_HEIGHT, SCALE_PERCENT, SAFE_DISTANCE, FOCAL_LENGTH]
```

Explanations for the parameters are the same as above in [Distance Evaluation on Annotated Data](#distance-evaluation-on-annotated-data). Note that SENSOR_WIDTH and SENSOR_HEIGHT parameters are mandatory, with FOCAL_LENGTH being optional. Usually, focal length information is included in the metadata of each image. If this is not the case for the images you are evaluating, please also provide FOCAL_LENGTH as a parameter. SENSOR_WIDTH, SENSOR_HEIGHT and FOCAL_LENGTH parameters are all in mm units and they apply to all of the images in PATH_TO_IMAGES. This means that all of the images under PATH_TO_IMAGES must have the same sensor width and sensor height and also same focal length only if FOCAL_LENGTH is provided. If FOCAL_LENGTH is not provided, the focal length information must be present in the metadata of each image and in that case, it is allowed to have different images with different focal lengths under PATH_TO_IMAGES. If you wish to evaluate images with different sensor dimensions, please categorize them by their sensor dimensions and put them under separate folders. Then, run the code for each separate folder.

### Output format
The outputs of the evaluation will automatically be written in a csv. file with a unique timestamp name under the folder "outputs_unlabeled_data". The naming format of the files are: "YYYYMMDD-HHMMSS.csv". The output file contains the following information for each image: filename, number of detected people, every estimated pair-wise distance, average estimated pair-wise distance and number of violations. Finally, the last row with filename 'ALL' contains the average of each column. All of the units are in mm.

## Adding Your Own Annotated Images

It is possible to add your own images to the dataset. Please follow the following annotation format and add the necessary information for each image in all three .csv files under the folder 'labels'. Illustration of how each of these files should look like can be seen below:

<img src="https://user-images.githubusercontent.com/79134040/110604226-81546e80-8190-11eb-94fa-86dbe12331ec.png" width="600">

New photo shoots, i.e., new settings of people, must be identified with a unique integer identifier. For any photo shoot, the real world locations of the people should stay the same in all the photos. There may be pictures taken from different camera locations. Person and camera tags should start with a P and C letter, respectively, followed by a unique identifier integer. Person and camera location tags must be consistent within a given photo shoot, however repeated tags in different photo shoots are allowed. This means that two different people or camera tags could be the same as long as they belong to a different photo shoot. At least 1 of 4 body parts (center of the eyes, shoulders, torso, head) of the people in the images must be annotated in terms of pixel locations.

After adding the annotations to the three .csv files under the folder 'labels' by following this format, you can follow the directions in [Distance Evaluation on Annotated Data](#distance-evaluation-on-annotated-data) to evaluate your images.


## Testing Your Own Method

If you have your own method that can output 3D location estimations for the people or the distance between the people in images, you may evaluate your method by using our dataset (download link in [Dataset and Annotations](#dataset-and-annotations)). You may also add your own images for evaluation, but be sure to correctly annotate each image by following the directions in  [Adding Your Own Annotated Images](#adding-your-own-annotated-images). Your method should be able to provide the pixel locations of at least 1 of the 4 following body parts for each detected person: center of eyes, torso, shoulders or head. These pixel locations are required in order to match the automatically detected people with the annotated people. Please note that if you wish to evaluate your method on your own added images, make sure that your method can output at least 1 of the body parts that you annotated. As an example, if you only annotated the center of torsos and your method only outputs the center of heads, the code will not work.

All of the necessary code for automatically evaluating your own method on our dataset or on your own annotated images can be found in the file 'automatic_evaluation_API.py'.

If your method outputs 3D location estimations for each detected people, you may use the function *automatic_evaluate()* and if your method outputs the distances between people, you may use the function *automatic_evaluate_2()*. Further explanations and example use cases are included within the file 'automatic_evaluation_API.py'.

## Results

Out of all the detected people, our method was able to achieve a 28.8% average error in pair-wise distance estimations when we evaluated it on our own annotated dataset. Furthermore, 92% of the people in the images were detected. 

## References

- YOLOv4: https://github.com/AlexeyAB/darknet
- OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose
