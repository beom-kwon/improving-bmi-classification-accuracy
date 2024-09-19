# Contents
1. [Introduction](#introduction)
2. [How to Run](#how-to-run)
3. [Citation](#citation)
4. [License](#license)

# Introduction

On this web page, we provide the Python implementation of the body mass index (BMI) classification method proposed in our paper titled '[Improving BMI Classification Accuracy with Oversampling and 3-D Gait Analysis on Imbalanced Class Data](https://doi.org/10.9708/jksci.2024.29.09.055).' In this study, we proposed a method to improve the classification accuracy of BMI estimation techniques based on three-dimensional gait data. Additionally, we demonstrated the usefulness of anthropometric and spatiotemporal features in gait data-based BMI estimation techniques. The experimental results showed that using both features together and applying an oversampling technique achieves state-of-the-art performance with 92.92% accuracy in the BMI estimation problem.

# How to Run

## 1. Dataset Preparation

* We utilized the Kinect Gait Biometry Dataset which consists of data from 164 individuals walking in front of an Xbox 360 Kinect Sensor.
* If you wish to download this dataset, please click [here](https://www.researchgate.net/publication/275023745_Kinect_Gait_Biometry_Dataset_-_data_from_164_individuals_walking_in_front_of_a_X-Box_360_Kinect_Sensor) to access the download link.
* To obtain BMI information, you will also need to download the Gender and Body Mass Index (BMI) Data for the Kinect Gait Biometry Dataset.
* Please click [here](https://www.researchgate.net/publication/308929259_Gender_and_Body_Mass_Index_BMI_Data_for_Kinect_Gait_Biometry_Dataset_-_data_from_164_individuals_walking_in_front_of_a_X-Box_360_Kinect_Sensor) to access the download link for the gender data.

## 2. Data Transformation (from .txt to .npz)

The skeleton sequences in the Kinect Gait Biometry Dataset are stored as text files with the filename extension .txt. To facilitate data handling, we read all gait sequences in .txt format and then saved them into a single file in uncompressed .npz format. Please run 'step1_data_transformation_from_txt_to_npz.py.' After running the .py file, you will obtain an .npz file.

## 3. Feature Extraction

Please execute 'step2_feature_extraction.py.' After running the .py file, you will obtain three .npz files containing feature vectors.

## 3. Performance Evaluation 

Please execute 'step3_performance_evaluation.py.' Running this .py file will display the evaluation results for each machine learning model.

# Citation

Please cite the following papers in your publications if they contribute to your research.

```
@article{kwon2020ensemble,
  author={Kwon, Beom and Lee, Sanghoon},
  journal={Applied Sciences},
  title={Ensemble Learning for Skeleton-Based Body Mass Index Classification},  
  year={2020},
  volume={10},
  number={21},
  pages={1--23},  
  doi={10.3390/app10217812}
}
```
Paper link: [Ensemble Learning for Skeleton-Based Body Mass Index Classification](https://doi.org/10.3390/app10217812)

```
@article{kwon2024improving,
  author={Kwon, Beom},
  journal={Journal of The Korea Society of Computer and Information},
  title={Improving BMI Classification Accuracy with Oversampling and 3-D Gait Analysis on Imbalanced Class Data},
  year={2024},
  volume={29},
  number={9},
  pages={55--66},
  doi={10.9708/jksci.2024.29.09.055}
}
```
Paper link: [Improving BMI Classification Accuracy with Oversampling and 3-D Gait Analysis on Imbalanced Class Data](https://doi.org/10.9708/jksci.2024.29.03.055)

# License

Our codes are freely available for non-commercial use.
