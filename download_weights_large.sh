mkdir -p logs
cd logs
echo "File size is 712 MB"
gdown https://drive.google.com/uc?id=1J5LAnoxG_BCGanC9vObY5ETSrRH47B3_
tar -zxvf checkpoints.tar.gz
rm -rf checkpoints.tar.gz
