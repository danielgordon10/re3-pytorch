mkdir -p logs
cd logs
echo "File size is 440 MB"
gdown https://drive.google.com/uc?id=1IEZKqee75EeX1K1aUvfZnE0KOZ44YHQf
tar -zxvf checkpoints_small.tar.gz
rm -rf checkpoints_small.tar.gz
