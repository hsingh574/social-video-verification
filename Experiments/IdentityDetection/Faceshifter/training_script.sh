sudo docker run -itd --ipc host --gpus all -v /media/eleanor/New-Volume/faceshifter:/workspace -v /media/eleanor/New-Volume/faceshifter:/DATA --name FS -t faceshifter:0.0
sudo docker attach FS
python aei_trainer.py -c config/train.yaml -n faceshifter
