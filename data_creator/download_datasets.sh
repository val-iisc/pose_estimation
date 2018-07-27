echo "Downloading Pascal 3D+ dataset"
wget ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip -O $data_dir/pascal3d.zip
echo "Unzipping the dataset"
unzip $data_dir/pascal3d.zip -d $data_dir
rm $data_dir/pascal3d.zip


echo "Downloading ObjectNet3D dataset"
wget ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_images.zip -O $data_dir/objectnet3d.zip
echo "Unzipping the dataset"
unzip $data_dir/objectnet3d.zip -d $data_dir
rm $data_dir/objectnet3d.zip
# Need to unzip the annoatations
echo "Downloading Objectnet3D Annotations"
wget ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_annotations.zip -O $data_dir/objectnet3d.zip
echo "Unzipping the annotations"
unzip $data_dir/objectnet3d.zip -d $data_dir/ObjectNet3D/
rm $data_dir/objectnet3d.zip
echo "Downloading Objectnet3D indexes"
wget ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_image_sets.zip -O $data_dir/objectnet3d.zip
echo "Unzipping the indexes"
unzip $data_dir/objectnet3d.zip -d $data_dir/ObjectNet3D/
rm $data_dir/objectnet3d.zip


echo "Downloading keypoint 5 dataset"
wget http://3dinterpreter.csail.mit.edu/data/keypoint-5.zip -O $data_dir/keypoint-5.zip
echo "Unzipping the dataset"
mkdir $data_dir/keypoint-5 
unzip $data_dir/keypoint-5.zip -d $data_dir/keypoint-5/
rm $data_dir/keypoint-5.zip


# IKea and VKITTI to be added.
