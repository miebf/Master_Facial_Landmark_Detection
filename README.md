# Lightweight ADNet for Efficient and Accurate Facial Landmark Detection on Resource-Constrained Devices


##Directory structure

| Folder/File         | Description                                           |
|---------------------|-------------------------------------------------------|
| conf                | Configure files for model training                    | 
| lib                 | Core file                                             | 
| evaluate.py         | EScript for model evaulation                          | 
| main.py             | Entry script for training and resting                 | 
| onnxQuant.py        | File for making a  model.onnx into a PTQ model.onnx   | 
| requirements.txt    | The dependency list                                   | 
| tester.py           | Script for model testing                              | 
| trainer.py          | Main script for model training                        | 

##Dependencies
Local:

- To run on GPU you need to install CUDA (What version depends on which GPU you have)
- requirements.txt
- Torch (Version depends on Cuda version)

##Preparation
https://www.kaggle.com/datasets/toxicloser/ibug-300w-large-face-landmark-dataset
- Clone repo
- Install dependencies
- Downlaod the raw images from datasets [COFW](https://data.caltech.edu/records/bc0bf-nc666),[WFLW](https://wywu.github.io/projects/LAB/WFLW.html)) and [300W](https://ibug.doc.ic.ac.uk/resources/300-W/) (If official site doesn't work for 300W, you can use [kaggle 300W](https://www.kaggle.com/datasets/toxicloser/ibug-300w-large-face-landmark-dataset))
- This repo follows the same data preprocess as in [ADNet](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_ADNet_Leveraging_Error-Bias_Towards_Normal_Direction_in_Face_Alignment_ICCV_2021_paper.pdf) so the metadata can be downloaded from [their repository](https://github.com/huangyangyu/ADNet/tree/main)
- Put the metadata and dataset images into data/alignment/${dataset} folder
- The dataset directory should look like this:
```text project-root/ 
├──data
     └──alignment
             ├──WFLW
                  ├──rawImages  ---WLFW_images
                  ├──train.tsv
                  └──test.tsv
             ├──COFW
                  ├──rawImages  ---COFW_images
                  ├──train.tsv
                  └──test.tsv
             └──300W
                  ├──rawImages  ---300W images
                  ├──train.tsv
                  └──test.tsv
```

##Training
```text
python main.py --mode=train --config_name=alignment --device_ids=0,1,2,3
```
##Testing
```text
python main.py --mode=test --config_name=alignment --pretrained_weight=${model_path} --device_ids=0
```
##Evaluation
```text
python evaluate.py --mode=nme --config_name=alignment --model_path=${model_path} --metadata_path==${metadata_path} --image_dir=${image_dir} --device_ids=0
```


##Acknowledgments
This repository is built on top of [ADNet](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_ADNet_Leveraging_Error-Bias_Towards_Normal_Direction_in_Face_Alignment_ICCV_2021_paper.pdf).


  
