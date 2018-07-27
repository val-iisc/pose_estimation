## Render for CNN: *Viewpoint Estimation in Images Using CNNs Trained with Rendered 3D Model Views*
Created by <a href="http://ai.stanford.edu/~haosu/" target="_blank">Hao Su</a>, <a href="http://web.stanford.edu/~rqi/" target="_blank">Charles R. Qi</a>, <a href="http://web.stanford.edu/~yangyan/" target="_blank">Yangyan Li</a>, <a href="http://geometry.stanford.edu/member/guibas/" target="_blank">Leonidas J. Guibas</a> from Stanford University.

## Modified RenderForCNN
Modified by <a href="http://bardofcodes.github.io" target="_blank">Aditya Ganeshan</a>. Basic Changes:

1) Depth render and seperate alpha map saving
2) saving crop parameters
3) Uniform sampling of only a single template model per-class (~8k Images per class).

### Introduction

Hao Su etal.'s work was initially described in an [arXiv tech report](http://arxiv.org/abs/1505.05641) and appeared as an ICCV 2015 paper. Render for CNN is a scalable image synthesis pipeline for generating millions of training images for high-capacity models such as deep CNNs. They demonstrated how to use this pipeline, together with specially designed network architecture, to train CNNs to learn viewpoints of objects from millions of synthetic images and real images. In this repository, we modifed the code for our research on pose estimation. Modified RenderForCNN was used for generating the synthetic template 3D model's render. Our work **iSPA-Net: Iterative Spatial Alignment Network** will appear as a ACM-MM 2018 paper. Have a look [here]()!


### License

Render for CNN is released under the MIT License (refer to the LICENSE file for details).


### Citing Render for CNN
If you find Render for CNN useful in your research, please consider citing:

    @InProceedings{Su_2015_ICCV,
        Title={Render for CNN: Viewpoint Estimation in Images Using CNNs Trained with Rendered 3D Model Views},
        Author={Su, Hao and Qi, Charles R. and Li, Yangyan and Guibas, Leonidas J.},
        Booktitle={The IEEE International Conference on Computer Vision (ICCV)},
        month = {December},
        Year= {2015}
    }


###  Render for CNN Image Synthesis Pipeline: Useful Original Instructions

**Prerequisites**

0. Blender (tested with Blender 2.71 on 64-bit Linux). You can get it from <a href="http://www.blender.org/features/past-releases/2-71/" target="_blank">Blender website</a> for free.


1. Datasets (ShapeNet, PASCAL3D+, SUN2012) [**not required for the demo**]. If you already have the same datasets (as in urls specified in the shell scripts) downloaded, you can build soft links to the datasets with the same pathname as specified in the shell scripts. Otherwise, just do the following steps under project root folder:
	
    <pre>
    bash dataset/get_shapenet.sh
    bash dataset/get_sun2012pascalformat.sh
    bash dataset/get_pascal3d.sh
    </pre>
    
**Set up paths**

All data and code paths should be set in `global_variables.py`. (currently BASEPATH to be set in `render_model_views.py`).

# Modified-RenderForCNN:

After setting up the file and location links in `global_variables.py`, you can simply run the following commands to generate the images:

```
# First set the variables in global_variables.py

# Now run render
python run_render.py

# Post rendering for resize, and alpha-map etc.
python preprocess_synthetic_images.py
```
