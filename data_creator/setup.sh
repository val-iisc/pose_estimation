data_dir=../data
pascal_dir=$data_dir/pascal3D
objectnet_dir=$data_dir/ObjectNet3D
keypoint5_dir=$data_dir/keypoint-5
anchors_dir=$data_dir/anchors

# for softlinks change below
soft_pascal=/data1/aditya/clean_content/data/pascal3D
soft_objectnet=/data1/aditya/clean_content/data/ObjectNet3D
soft_keypoint5=/data1/aditya/clean_content/data/keypoint-5

declare -A arr
arr["keypoint-5"]="chair sofa table bed swivelchair"
arr["pascal3D"]="chair_imagenet sofa_imagenet diningtable_imagenet chair_pascal sofa_pascal diningtable_pascal"
arr["ObjectNet3D"]="chair sofa diningtable bed"

# create the preprocessed_data folders

if [ "$1" = "make_save_dirs" ];
then
    mkdir $data_dir/image_lists
    for x in bbox dense_keypoints pose correspondence;do
        mkdir $data_dir/$x
    done
    mkdir $data_dir/real_data
    # for synthetic images
    mkdir $data_dir/synthetic_data
    for i in keypoint-5 pascal3D ObjectNet3D;do
        mkdir $data_dir/real_data/$i
        for j in ${arr[$i]};do
            mkdir $data_dir/real_data/$i/$j
            for k in images sparse_keypoints dense_keypoints;do
	        mkdir $data_dir/real_data/$i/$j/$k
            done
        done
    done
fi

if [ "$1" = "set_soft_links" ];
then
  ln -s $soft_pascal $pascal_dir
  ln -s $soft_objectnet $objectnet_dir
  ln -s $soft_keypoint5 $keypoint5_dir
fi

declare -A classArr
classArr['pascal3D']="chair sofa diningtable"
classArr['Objectnet3D']="chair sofa diningtable bed"
classArr['keypoint-5']="chair sofa table bed swivelchair"

declare -A subsetArr
subsetArr['pascal3D']="imagenet pascal"
subsetArr['ObjectNet3D']="nil"
subsetArr['keypoint-5']="nil"
# for generating all datasets (crop + sparse)
if [ "$1" = "generate_all_crops" ];
then


for i in keypoint-5 pascal3D ObjectNet3D;do
    for j in ${classArr[$i]};do
        for k in ${subsetArr[$i]};do
            echo "processing real files for $i dataset, class $j, subset $k"
            python preprocess_real_images.py --dataset $i --class $j --subset $k
        done
    done
done

    
fi

# for generating all dense keypoints
if [ "$1" = "generate_all_keypoints" ];
then

for i in keypoint-5 pascal3D ObjectNet3D;do
    for j in ${classArr[$i]};do
        for k in ${subsetArr[$i]};do
            echo "processing real files for $i dataset, class $j, subset $k"
            python make_dense_keypoints.py --dataset $i --class $j --subset $k
        done
    done
done

fi

# for generating all subsets
if [ "$1" = "generate_all_subsets" ];
then

for i in keypoint-5 pascal3D ObjectNet3D;do
    for j in ${classArr[$i]};do
        for k in ${subsetArr[$i]};do
            echo "processing real files for $i dataset, class $j, subset $k"
            python make_subsets.py --dataset $i --class $j --subset $k
        done
    done
done

fi

