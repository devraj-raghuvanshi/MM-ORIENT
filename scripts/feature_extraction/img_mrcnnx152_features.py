import os
home = "/content"
os.chdir(home)
os.getcwd()

# Install specified versions of `torch` and `torchvision`, before installing mmf (causes an issue)
!pip install torch==1.6.0 torchvision==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# Clone the following repo where mmf does not install default image features,
# since we will use our own features
!git clone --branch no_feats --config core.symlinks=true https://github.com/rizavelioglu/mmf.git

os.chdir(os.path.join(home, "mmf"))

!pip install --editable .

PATH_TO_ZIP_FILE = "/full_path_to_the_zip_file/memotion.zip"
!cp $PATH_TO_ZIP_FILE /content/mmf/

# Add the mmf folder to Python Path
os.environ['PYTHONPATH'] += ":/content/mmf/"

!mmf_convert_hm --zip_file="memotion.zip"

import os
os.chdir(home)
!git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark

!pip install ninja yacs cython matplotlib

os.chdir(os.path.join(home, "vqa-maskrcnn-benchmark"))
!rm -rf build
!python setup.py build develop

# !wget https://dl.fbaipublicfiles.com/pythia/detectron_model/FAST_RCNN_MLP_DIM2048_FPN_DIM512.pkl
# !wget https://dl.fbaipublicfiles.com/pythia/detectron_model/e2e_faster_rcnn_X-101-64x4d-FPN_1x_MLP_2048_FPN_512.yaml
os.chdir(os.path.join(home, "mmf/tools/scripts/features/"))
out_folder = os.path.join(home, "features/")

!python extract_features_vmb.py --config_file "https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model_x152.yaml" \
                                --model_name "X-152" \
                                --output_folder $out_folder \
                                --image_dir "/root/.cache/torch/mmf/data/datasets/memotion/images/" \
                                --num_features 100 \
                                # --exclude_list "/content/exclude.txt"
                                # --feature_name "fc6" \
                                # --confidence_threshold 0. \