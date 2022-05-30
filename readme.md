# 环境配置
## 讲义
* https://github.com/megvii-research/introduction-neural-3d-reconstruction
## Install
```bash
pip install -r requirements.txt
```
### 建议版本
Tested on Ubuntu with torch 1.10 & CUDA 11.3 on TITAN RTX & python 3.7.11

Currently, `--ff` only supports GPUs with CUDA architecture `>= 70`.

# torch-ngp
- 接入数据集，当前仅支持`kitti_odometry`数据集，首先在当前目录下新建`datasets`文件夹:
    - `kitti_odometry`数据可以通过[链接](https://studio.brainpp.com/dataset/3840?name=kiiti%EF%BC%88%E5%A4%84%E7%90%86%E8%BF%87%E7%9A%84%E9%83%A8%E5%88%86%E6%95%B0%E6%8D%AE%EF%BC%89)获得。下载数据，解压后将整个文件夹拷贝到`datasets`目录下
    - 如果需要其他场景的数据，请将数据集按照`kitti_odometry`数据集的格式拷贝到`datasets`文件夹下，并且修改对应的config
    - 提示：所有数据都放进去直接训练很可能会OOM，作业要求是对场景分块，想要快速测试网络，可以将`./nerf/Atlantic_datasets/selector.py`里面选择的图片数量改少一点（不能过于少，因为数据处理里面有对于整个场景的归一化操作，过于少无法选择合适的框，建议选择50张图片）
    - 其他：数据集包含pose和image。pose中前面部分和image有序对应。多余的pose可以用做测试渲染新视角的图片（PS：如果是训练场景外的pose，渲染效果差于场景内），也可以选择不用这部分pose，自定义渲染视角测试效果。

- 训练脚本改为通过config配置
    - 如果不指定config文件则会按照默认的配置执行
    - config文件按照：基础配置，数据，网络参数等分级配置
    - 可以通过修改config配置，执行不同的实验
 
- exercise 相关：
    - 实现“大场景”（包含数据集1600+图片内容的场景）分块融合
    - 本代码只提供训练数据集的前100张图片，仅作为codebase，分块和融合策略请自行添加
    - 提交代码压缩包和可视化效果（图片视频）
        - 可以额外附加文档说明分块融合的策略（也可以写在代码注释中）
        - 可视化不仅要有训练集中的pose渲染图片，还要有val集的pose渲染效果，最好还有其他视角（插值/自定义）的渲染效果（酌情加分）

## Train
```bash
python ./train_nerf.py -c configs/kitti/kitti_00.yaml
```
## Test
```bash
python ./test_nerf.py -c configs/kitti/kitti_00.yaml
```

若ffmpeg报错，建议通过conda重装ffmpeg

---
# 代码说明
A pytorch implementation of [instant-ngp](https://github.com/NVlabs/instant-ngp), as described in [_Instant Neural Graphics Primitives with a Multiresolution Hash Encoding_](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf).


# Difference from the original implementation
* Instead of assuming the scene is bounded in the unit box `[0, 1]` and centered at `(0.5, 0.5, 0.5)`, this repo assumes **the scene is bounded in box `[-bound, bound]`, and centered at `(0, 0, 0)`**. Therefore, the functionality of `aabb_scale` is replaced by `bound` here.
* For the hashgrid encoder, this repo only implement the linear interpolation mode.
* For the voxel pruning in ray marching kernels, this repo doesn't implement the multi-scale density grid (check the `mip` keyword), and only use one `128x128x128` grid for simplicity. Instead of updating the grid every 16 steps, we update it every epoch, which may lead to slower first few epochs if using `--cuda_ray`.
* For the blender dataest, the default mode in instant-ngp is to load all data (train/val/test) for training. Instead, we only use the specified split to train in CMD mode for easy evaluation. However, for GUI mode, we follow instant-ngp and use all data to train (check `type='all'` for `NeRFDataset`).


# Acknowledgement

* Credits for the amazing [torch-ngp](https://github.com/ashawkey/torch-ngp), which provides a pytorch CUDA extension implementation of instant-ngp (sdf and nerf). I am extremely grateful to [Ashawkey](https://github.com/ashawkey) open source this rewarding code. With this torch version codebase, it is easier to understand instnt-ngp and more convenient to experiment iteration.

* Difference from the original torch-ngp:
    * 保持：
        * 除nerf文件夹以及`train_nerf.py, test_nerf.py`以外的其他文件和文件夹都保持不变
    * 增加：
        * 场景适配性，以及数据处理的代码
        * `depth，seg`等监督，以及新增采样策略
    * 修改：
        * 拆分重建了一些原有结构体
        * 优化网络结构

* The framework of NeRF is adapted from [nerf_pl](https://github.com/kwea123/nerf_pl):
    ```
    @misc{queianchen_nerf,
        author = {Quei-An, Chen},
        title = {Nerf_pl: a pytorch-lightning implementation of NeRF},
        url = {https://github.com/kwea123/nerf_pl/},
        year = {2020},
    }
    ```
