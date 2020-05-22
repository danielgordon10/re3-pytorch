mkdir -p logs
cd logs
echo "File size is 712 MB"
gdown https://drive.google.com/uc?id=1EmA3fnPh1tQXMsZtsJo3YkGw93Ytz8Hd
tar -zxvf checkpoints.tar.gz
rm -rf checkpoints.tar.gz
