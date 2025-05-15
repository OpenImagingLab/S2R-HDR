##  S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion
### [Project Page](https://openimaginglab.github.io/S2R-HDR/) | [ArXiv](https://arxiv.org/abs/2504.07667) | [Dataset (part1)](https://huggingface.co/datasets/iimmortall/S2R-HDR) [Dataset (part2)](https://huggingface.co/datasets/iimmortall/S2R-HDR-2) <br>

Yujin Wang, Jiarui Wu, Yichen Bian, Fan Zhang, Tianfan Xue<br><br>


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

- Downlaod S2R-HDR dataset from Hugging Face.

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
                    local_dir='./data/S2R-HDR', # setting save path
                    resume_download=True)
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
    ``` shell
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
-- 
## 🛠️ Training
### Training model on S2R-HDR datasets.
- Training SCTNet model on S2R-HDR datasets.
    ``` shell
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

## 🛠️ Testing
### Adapt to real capture datasets with ground-truth.
1. Using real capture datasets with ground-truth to train S2R-Adapter network.
    - Adapt to SCT dataset
        ``` shell
        # 8 GPU
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
            --batch_size=6 --num_workers=6 --lr 0.0002 --lr_min 0.0002 --test_interval 1
        ```  
    - Adapt to Chalenge123 dataset
        ``` shell
        # 8 GPU
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
            --batch_size=4 --num_workers=4 --lr 0.0002 --lr_min 0.0002 --test_interval 1
        ```
2. Testing adapted models on testing dataset.
    - Adapt to SCT dataset
        ``` shell
        CUDA_VISIBLE_DEVICES=0 python test_adapter_with_gt.py --save_results --model SCTNet \
            --pretrained_model pretrained_models/sctnet-adapter-with-gt-sct.pth \
            --save_dir experiments/sctnet-adapter-with-gt-sct/results_ada_sct_test_sct \
            --dataset_dir data/sct
        ```  
    - Adapt to Chalenge123 dataset
        ``` shell
        CUDA_VISIBLE_DEVICES=0 python test_adapter_with_gt.py --save_results --model SCTNet \
          --pretrained_model pretrained_models/sctnet-adapter-with-gt-challenge123.pth \
          --save_dir experiments/sctnet-adapter-with-gt-cha/results_ada_cha_test_cha \
          --dataset_dir data/challenge123 \
          --data_name challenge123
        ```
### Adapt to real capture datasets without ground-truth.
``` shell

```


## Citation

Please cite us if our work is useful for your research.
```
@article{wang2025s2r,
  title={S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion},
  author={Wang, Yujin and Wu, Jiarui and Bian, Yichen and Zhang, Fan and Xue, Tianfan},
  journal={arXiv preprint arXiv:2504.07667},
  year={2025}
}
```

## Acknowledgements

This project is based on [Unreal Engine5](https://www.unrealengine.com/en-US/unreal-engine-5), [HDR-Transformer](https://github.com/liuzhen03/HDR-Transformer-PyTorch), [SCTNet](https://github.com/Zongwei97/SCTNet), and [SAFNet](https://github.com/ltkong218/SAFNet). Thanks for their awesome work.
