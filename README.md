# Facial Keypoint detection using PyTorch

### Local Environment Instructions

    conda create -n cv-nd python=3.6
    source activate cv-nd
    conda install pytorch torchvision cuda91 -c pytorch
    pip install -r requirements.txt

## Download the data

    wget -P ./data/ https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aea1b91_train-test-data/train-test-data.zip
    cd data
    unzip train-test-data.zip
    rm train-test-data.zip