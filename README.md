<p align="center">

  <h2 align="center">
  S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion
  
  (ICLR 2026)
  </h2>
  <p align="center">
    <a><strong>Yujin Wang*</strong></a><sup>1</sup>
    .
    <a><strong>Jiarui Wu*</strong></a><sup>2,1</sup>
    .
    <a><strong>Yichen Bian*</strong></a><sup>1,4</sup>
    .
    <a><strong>Fan Zhang</strong></a><sup>1</sup>
    ·
    <a href="https://tianfan.info/"><strong>Tianfan Xue</strong></a><sup>2,1,3</sup>
    <!-- <br> -->
    <br>
    <sup>1</sup>Shanghai AI Laboratory, <sup>2</sup>CUHK MMLab, 
    <sup>3</sup>CPII under InnoHK, <sup>4</sup>Shanghai Jiao Tong University
    <br>
    <div align="center">
    <a href="https://arxiv.org/abs/2504.07667"><img src='https://img.shields.io/badge/arXiv-S2R_HDR-red' alt='Paper PDF'></a>
    <a href='https://openimaginglab.github.io/S2R-HDR/'><img src='https://img.shields.io/badge/Project_Page-S2R_HDR-blue' alt='Project Page'></a>
    <a href='https://huggingface.co/datasets/iimmortall/S2R-HDR'><img src='https://img.shields.io/badge/Datasets-%F0%9F%A4%97%20Hugging%20Face-yellow'></a>
    <a href='https://www.kaggle.com/datasets/iimmortall/s2r-hdr'><img src='https://img.shields.io/badge/Datasets-Kaggle-green'></a>
    </div>
  </p>
</p>

## ✨Dataset Summary

S2R-HDR is a large-scale synthetic dataset for high dynamic range (HDR) reconstruction tasks.  It contains 1,000 motion sequences, each comprising 24 images at 1920×1080 resolution, with a total of **24,000 images**.  To support flexible data augmentation, all images are stored in **EXR format with linear HDR values**.  The dataset is rendered using Unreal Engine 5 and our custom pipeline built upon XRFeitoria, encompassing diverse dynamic elements, motion patterns, HDR-compatible scenes, and varied lighting conditions.  Beyond the core imagery, we additionally provide per-frame rendered auxiliary data including optical flow, depth maps, surface normals, and diffuse albedo information, significantly expanding S2R-HDR's potential applications across various computer vision tasks.

<table>
    <tr>
      <td ><center><img src="https://i.postimg.cc/B6cR10Mj/1-img.gif" ></center></td>
      <td ><center><img src="https://i.postimg.cc/rF763cZr/1.gif"  ></center></td>
      <td ><center><img src="https://i.postimg.cc/J73RbwGc/1-depth.gif"  ></center></td>
      <td ><center><img src="https://i.postimg.cc/DwSVpRyN/1-diffuse.gif"  ></center></td>
      <td ><center><img src="https://i.postimg.cc/cHNVHHMq/1-normal.gif"  ></center></td>
    </tr>
    <tr>
      <td ><center><img src="https://i.postimg.cc/nr1s5CTn/2-img.gif" ></center></td>
      <td ><center><img src="https://i.postimg.cc/zGWm1M4P/2.gif"  ></center></td>
      <td ><center><img src="https://i.postimg.cc/W4ZBTrZ6/2-depth.gif"  ></center></td>
      <td ><center><img src="https://i.postimg.cc/mkLJY558/2-diffuse.gif"  ></center></td>
      <td ><center><img src="https://i.postimg.cc/kgGV1MSd/2-normal.gif"  ></center></td>
    </tr>
    <tr>
      <td ><center><img src="https://i.postimg.cc/ZYjKGt4Y/3-img.gif" ></center></td>
      <td ><center><img src="https://i.postimg.cc/bvHKZnyx/3.gif"  ></center></td>
      <td ><center><img src="https://i.postimg.cc/gJHkc1hD/3-depth.gif"  ></center></td>
      <td ><center><img src="https://i.postimg.cc/qqBBNQXd/3-diffuse.gif"  ></center></td>
      <td ><center><img src="https://i.postimg.cc/7Y6YDsZ3/3-normal.gif"  ></center></td>
    </tr>
    <tr>
        <td ><center>HDRs</center></td>
        <td ><center>Optical Flow</center></td>
        <td ><center>Depth</center></td>
        <td ><center>Diffuse</center></td>
        <td ><center>Normal</center></td>
    </tr>
</table>

Note: 
- The S2R-HDR dataset on HuggingFace is stored in two separate repository: 

    https://huggingface.co/datasets/iimmortall/S2R-HDR

    https://huggingface.co/datasets/iimmortall/S2R-HDR-2.

- The S2R-HDR dataset on Kaggle is stored in four separate repository: 
    
    https://www.kaggle.com/datasets/iimmortall/s2r-hdr

    https://www.kaggle.com/datasets/iimmortall/s2r-hdr-2

    https://www.kaggle.com/datasets/iimmortall/s2r-hdr-3

    https://www.kaggle.com/datasets/iimmortall/s2r-hdr-4


## 🛠️ Installation
1. Install Anaconda or Miniconda.
2. Create a new environment.
   - Using the provided environment.yml file.
       ```shell
       conda env create -f environment.yml
       ```
   - Or using the provided requirements.txt file.
       ```shell
       conda create -n s2r python=3.12.7
       conda activate s2r
       pip install -r requirements.txt
       ```
3. Activate environment.
   ```shell
   conda activate s2r
   pip install huggingface_hub kagglehub
   ```
4. Download code.
    `https://github.com/OpenImagingLab/S2R-HDR.git`

## 🛠️ Prepare Dataset
Download all dataset to `./data` path.

- Downlaod S2R-HDR dataset from Hugging Face (Recommend).

    ``` python
    from huggingface_hub import snapshot_download
    # download part1
    snapshot_download(repo_id='iimmortall/S2R-HDR',
                    repo_type="dataset",
                    local_dir='./data/S2R-HDR', # setting save path
                    resume_download=True)
    # download part2
    snapshot_download(repo_id='iimmortall/S2R-HDR-2',
                    repo_type="dataset",
                    local_dir='./data/S2R-HDR-2', # setting save path
                    resume_download=True)
    ```
    ```shell
    mv ./data/S2R-HDR-2/* ./data/S2R-HDR/
    ```
- Downlaod S2R-HDR dataset from Kaggle, [kagglehub guidelines](https://github.com/Kaggle/kagglehub).
    ``` python
    import kagglehub
    # Download latest version
    path1 = kagglehub.dataset_download("iimmortall/S2R-HDR")
    print("Path1 to dataset files:", path)
    # mv path1/* ./data/S2R-HDR
    path2 = kagglehub.dataset_download("iimmortall/S2R-HDR-2")
    print("Path2 to dataset files:", path)
    # mv path2/* ./data/S2R-HDR
    path3 = kagglehub.dataset_download("iimmortall/S2R-HDR-3")
    print("Path3 to dataset files:", path)
    # mv path3/* ./data/S2R-HDR
    path4 = kagglehub.dataset_download("iimmortall/S2R-HDR-4")
    print("Path4 to dataset files:", path)
    # mv path4/* ./data/S2R-HDR
    ```
- Downlaod [SCT](https://drive.google.com/drive/folders/1CtvUxgFRkS56do_Hea2QC7ztzglGfrlB) dataset and [Challenge123](https://huggingface.co/datasets/ltkong218/Challenge123).
    
    Modify the hdr filename of challenge123 dataset, specifically, hdr_img.hdr -> HDRImg.hdr.
    ```shell 
    cd /xxxx/challenge123 # cd challenge123 data path
    pwd
    find . -type f -name "hdr_img.hdr" | while read -r file; do
        dir=$(dirname "$file")
        new_file="$dir/HDRImg.hdr"
        mv "$file" "$new_file" && echo "rename $file -> $new_file" 
    done
    echo "successful"
    ```
- All data structure:
    ```
    - S2R-HDR
        - scene_0_FM
            - camera_params
            - img
                - 0000.exr
                - [...]
                - 0023.exr
            - diffuse
                - [...]
            - flow
                - [...]
            - depth
                - [...]
            - normal
                - [...]
    - sct
        - Training/Test
            - scene_xxx_x
                - input_1.tif
                - input_2.tif
                - input_3.tif
                - exposure.txt
                - HDRImg.tif
    - challenge123
        - Training/Test
            - xxx_1
                - ldr_img_1.tif
                - ldr_img_2.tif
                - ldr_img_3.tif
                - exposure.txt
                - HDRImg.tif
    ```
- Preprocess S2R-HDR dataset.
    ```python preprocess_s2r_hdr_data.py```

## 🛠️ Training
### Training model on S2R-HDR datasets.
- Training SCTNet model on S2R-HDR datasets.
    ```shell
    # 8GPU
    accelerate launch --multi_gpu --num_processes=8 \
      train.py --model SCTNet \
      --data_name s2r-hdr --dataset_dir data/S2R-HDR-processed-patch \
      --test_dataset_dir data/sct \
      --logdir experiments/sctnet-s2r-hdr \
      --batch_size=8 --num_workers=8 --lr 0.0002 --test_interval 10
    ```
- Training SAFNet model on S2R-HDR datasets.
    ``` shell
    # 8GPU
    accelerate launch --multi_gpu --num_processes=8 \
      train.py --model SAFNet \
      --data_name s2r-hdr --dataset_dir data/S2R-HDR-processed-patch \
      --test_dataset_dir data/challenge123 \
      --logdir experiments/ctnet-s2r-hdr \
      --batch_size=6 --num_workers=6 --lr 0.0002 --test_interval 10
    ``` 

## 🛠️ Pretrained Models
Please download this model to ./pretrained_models/ .
| Model Name | Training Data | Adapter Data| Link |
|------------|---------------|-------------|------|
| SCTNet     | S2R-HDR | - | [sctnet-s2r-hdr.pth](https://github.com/OpenImagingLab/S2R-HDR/releases/download/v1.0/sctnet-s2r-hdr.pth) |
| SAFNet     | S2R-HDR | - | [safnet-s2r-hdr.pth](https://github.com/OpenImagingLab/S2R-HDR/releases/download/v1.0/safnet-s2r-hdr.pth) |
| SCTNet     | S2R-HDR | SCT | [sctnet-adapter-with-gt-sct.pth](https://github.com/OpenImagingLab/S2R-HDR/releases/download/v1.0/sctnet-adapter-with-gt-sct.pth) |
| SCTNet     | S2R-HDR | Challenge123 | [sctnet-adapter-with-gt-challenge123.pth](https://github.com/OpenImagingLab/S2R-HDR/releases/download/v1.0/sctnet-adapter-with-gt-challenge123.pth) |
| SAFNet     | S2R-HDR | SCT | [safnet-adapter-with-gt-sct.pth](https://github.com/OpenImagingLab/S2R-HDR/releases/download/v1.0/safnet-adapter-with-gt-sct.pth) |
| SAFNet     | S2R-HDR | Challenge123 | [safnet-adapter-with-gt-challenge123.pth](https://github.com/OpenImagingLab/S2R-HDR/releases/download/v1.0/safnet-adapter-with-gt-challenge123.pth) |

## 🛠️ Testing
### Adapt to real capture datasets with ground-truth.
1. Using real capture datasets with ground-truth to train S2R-Adapter network.
    - Adapt to SCT training dataset with SCTNet (S2R-Adapter) method.
        ```shell
        # 4 GPU
        accelerate launch --multi_gpu --num_processes=8 \
            train_adapter_with_gt.py --model SCTNet \
            --data_name sct --dataset_dir data/sct \
            --test_dataset_dir data/sct \
            --logdir experiments/sctnet-s2r-hdr-adapter-sct \
            --resume pretrained_models/sctnet-s2r-hdr.pth \
            --epochs 30 \
            --lr_scale 1.0 \
            --r1 1 \
            --r2 64 \
            --learn_scale \
            --scale1 1.0 \
            --scale2 1.0 \
            --batch_size=4 --num_workers=4 --lr 0.0002 --lr_min 0.0001 --test_interval 1
        ```  
    - Adapt to Chalenge123 training dataset with SCTNet (S2R-Adapter) method.
        ```shell
        # 4 GPU
        accelerate launch --multi_gpu --num_processes=8 \
            train_adapter_with_gt.py --model SCTNet \
            --data_name challenge123-cache --dataset_dir data/challenge123 \
            --test_dataset_dir data/challenge123 \
            --logdir experiments/sctnet-s2r-hdr-adapter-cha \
            --resume pretrained_models/sctnet-s2r-hdr.pth \
            --epochs 30 \
            --lr_scale 1.0 \
            --r1 1 \
            --r2 64 \
            --learn_scale \
            --scale1 1.0 \
            --scale2 1.0 \
            --batch_size=4 --num_workers=4 --lr 0.0002 --lr_min 0.0001 --test_interval 1
        ```
    - Adapt to SCT training dataset with SAFNet (S2R-Adapter) method.
        ```shell
        # 4 GPU
        accelerate launch --multi_gpu --num_processes=8 \
            train_adapter_with_gt.py --model SAFNet \
            --data_name sct-cache --dataset_dir data/sct \
            --test_dataset_dir data/sct \
            --logdir experiments/safnet-s2r-hdr-adapter-sct \
            --resume pretrained_models/safnet-s2r-hdr.pth \
            --epochs 30 \
            --lr_scale 1.0 \
            --r1 1 \
            --r2 64 \
            --learn_scale \
            --scale1 1.0 \
            --scale2 1.0 \
            --batch_size=4 --num_workers=4 --lr 0.0002 --lr_min 0.0001 --test_interval 1
        ```  
    - Adapt to Chalenge123 training dataset with SAFNet (S2R-Adapter) method.
        ```shell
        # 4 GPU
        accelerate launch --multi_gpu --num_processes=8 \
            train_adapter_with_gt.py --model SAFNet \
            --data_name challenge123-cache --dataset_dir data/challenge123 \
            --test_dataset_dir ../../datasets/ImageHDR/challenge123 \
            --logdir experiments/safnet-s2r-hdr-adapter-cha \
            --resume pretrained_models/safnet-s2r-hdr.pth \
            --epochs 30 \
            --lr_scale 1.0 \
            --r1 1 \
            --r2 64 \
            --learn_scale \
            --scale1 1.0 \
            --scale2 1.0 \
            --batch_size=4 --num_workers=4 --lr 0.0002 --lr_min 0.0001 --test_interval 1
        ```
2. Testing adapted models on testing dataset.
    - Testing on SCT testing dataset with SCTNet (S2R-Adapter) method.
        ```shell
        CUDA_VISIBLE_DEVICES=0 python test_adapter_with_gt.py --save_results --model SCTNet \
            --pretrained_model pretrained_models/sctnet-adapter-with-gt-sct.pth \
            --save_dir experiments/sctnet-adapter-with-gt-sct/results_ada_sct_test_sct \
            --dataset_dir data/sct
        ```  
    - Testing on Chalenge123 testing dataset with SCTNet (S2R-Adapter) method.
        ```shell
        CUDA_VISIBLE_DEVICES=0 python test_adapter_with_gt.py --save_results --model SCTNet \
            --pretrained_model pretrained_models/sctnet-adapter-with-gt-challenge123.pth \
            --save_dir experiments/sctnet-adapter-with-gt-cha/results_ada_cha_test_cha \
            --dataset_dir data/challenge123 \
            --data_name challenge123
        ```
    - Testing on SCT testing dataset with SAFNet (S2R-Adapter) method.
        ```shell
        CUDA_VISIBLE_DEVICES=0 python test_adapter_with_gt.py --save_results --model SAFNet \
            --pretrained_model pretrained_models/safnet-adapter-with-gt-sct.pth \
            --save_dir experiments/safnet-adapter-with-gt-sct/results_ada_sct_test_sct \
            --dataset_dir data/sct
        ``` 
    - Testing on Chalenge123 testing dataset with SAFNet (S2R-Adapter) method.
        ```shell
        CUDA_VISIBLE_DEVICES=0 python test_adapter_with_gt.py --save_results --model SAFNet \
            --pretrained_model pretrained_models/safnet-adapter-with-gt-challenge123.pth \
            --save_dir experiments/safnet-adapter-with-gt-cha4/results_ada_cha_test_cha \
            --dataset_dir data/challenge123 \
            --data_name challenge123
        ```

### Adapt to real capture datasets without ground-truth.
Adapting to real capture datasets without ground truth during testing. In this example, we demonstrate how to adapt SAFNet to the SCT dataset, noting that ground truth information is unavailable to the model.
- Adapt to SCT testing dataset (without ground-truth) with SCTNet model.
    ```shell
    CUDA_VISIBLE_DEVICES=0 python train_adapter_without_gt.py --adapter --adaptive_scale --save_results --logdir experiments/TTA-SCTNet-sct/ \
        --data_name sct-cache \
        --model SCTNet \
        --dataset_dir data/sct \
        --test_dataset_dir data/sct \
        --test_interval 1 \
        --resume pretrained_models/sctnet-s2r-hdr.pth \
        --epochs 1 \
        --tta_aug_type 1 \
        --tta_img_num_patches 24 \
        --model_lr 0.0001 \
        --adapter_lr_scale 1 \
        --batch_size 1 \
        --model_ema_rate 0.999
    ```
- Adapt to Chalenge123 testing dataset (without ground-truth) with SCTNet model.
    ```shell
    CUDA_VISIBLE_DEVICES=0 python train_adapter_without_gt.py --adapter --adaptive_scale --save_results --logdir experiments/TTA-SCTNet-cha/ \
        --data_name sct-cache \
        --model SCTNet \
        --dataset_dir data/challenge123 \
        --test_dataset_dir data/challenge123 \
        --test_interval 1 \
        --resume pretrained_models/sctnet-s2r-hdr.pth \
        --epochs 1 \
        --tta_aug_type 1 \
        --tta_img_num_patches 24 \
        --model_lr 0.0001 \
        --adapter_lr_scale 1 \
        --batch_size 1 \
        --model_ema_rate 0.999
    ```
- Adapt to SCT testing dataset (without ground-truth) with SAFNet model.
    ```shell
    CUDA_VISIBLE_DEVICES=0 python train_adapter_without_gt.py --adapter --adaptive_scale --save_results --logdir experiments/TTA-SAFNet-sct/ \
        --data_name sct-cache \
        --model SAFNet \
        --dataset_dir data/sct \
        --test_dataset_dir data/sct \
        --test_interval 1 \
        --resume pretrained_models/safnet-s2r-hdr.pth \
        --epochs 1 \
        --tta_aug_type 1 \
        --tta_img_num_patches 24 \
        --model_lr 0.0001 \
        --adapter_lr_scale 1 \
        --batch_size 1 \
        --model_ema_rate 0.999
    ```
- Adapt to challenge123 testing dataset (without ground-truth) with SAFNet model.
    ```shell
    CUDA_VISIBLE_DEVICES=0 python train_adapter_without_gt.py --adapter --adaptive_scale --save_results --logdir experiments/TTA-SAFNet-cha/ \
        --data_name sct-cache \
        --model SAFNet \
        --dataset_dir data/challenge123 \
        --test_dataset_dir data/challenge123 \
        --test_interval 1 \
        --resume pretrained_models/safnet-s2r-hdr.pth \
        --epochs 1 \
        --tta_aug_type 1 \
        --tta_img_num_patches 24 \
        --model_lr 0.0001 \
        --adapter_lr_scale 1 \
        --batch_size 1 \
        --model_ema_rate 0.999
    ```

## Citation

Please cite us if our work is useful for your research.
```
@inproceedings{wang2025s2r,
  title={S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion},
  author={Wang, Yujin and Wu, Jiarui and Bian, Yichen and Zhang, Fan and Xue, Tianfan},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

## Acknowledgements

This project is based on [Unreal Engine5](https://www.unrealengine.com/en-US/unreal-engine-5), [xrfeitoria](https://github.com/openxrlab/xrfeitoria/), [HDR-Transformer](https://github.com/liuzhen03/HDR-Transformer-PyTorch), [SCTNet](https://github.com/Zongwei97/SCTNet), and [SAFNet](https://github.com/ltkong218/SAFNet). Thanks for their awesome work.
