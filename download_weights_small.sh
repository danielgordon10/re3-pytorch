mkdir -p logs
cd logs
echo "File size is 440 MB"
gdown https://drive.google.com/uc?id=1jvX4oxSe-N2CQGHUyFIvGodu7-Ox9m37
tar -zxvf checkpoints_small.tar.gz
rm -rf checkpoints_small.tar.gz
