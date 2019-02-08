# Objct_recognition
1) Open train.json and change paths to your system path
2) Divide training.csv into 2 parts, train.csv and train_val.csv
3) Save both in the working directory
4) Open image_provider.py and change paths appropriately to your system paths
5) To start training execute following commands 
````
export PYTHONPATH=$PWD
python train.py --exp train.json
````
