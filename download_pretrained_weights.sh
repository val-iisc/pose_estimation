gdown https://drive.google.com/uc?id=1-EHJCvQqb4UJQpHHlQ60IfPv2bB1sJq3
mv bvlc_googlenet.caffemodel ./ucn_code/nets/bvlc_googlenet/
gdown https://drive.google.com/uc?id=1zUOShxm0kLtd6uBxyk8c6xPumnszM8XF
unzip pretrained_weights.zip
rm -rf pretrained_weights.zip
mv pretrained_weights/* classifier_code/pretrained_weights/
rm -rf pretrained_weights
