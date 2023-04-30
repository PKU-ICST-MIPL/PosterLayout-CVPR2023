# PosterLayout: A New Benchmark and Approach for Content-aware Visual-Textual Presentation Layout

This repository contains the guidelines of benchmark **PKU PosterLayout** for "PosterLayout: A New Benchmark and Approach for Content-aware Visual-Textual Presentation Layout", CVPR 2023.

For **dataset details and downloads**, please visit our [project page](http://59.108.48.34/tiki/PosterLayout/).

<img src="/comparisons_vis.png" alt="comparisons_vis">
<p align="center">Comparison of layouts generated by different approaches.</p>

### Dataset
1. Download PKU PosterLayout from the [project page](http://59.108.48.34/tiki/PosterLayout/)
2. Unzip compressed files to corresponding directories
3. Put directories under ```Dataset/```, as follow:
```
Dataset/
├─ train/
│  ├─ inpainted_poster/
│  ├─ saliencymaps_basnet/
│  ├─ saliencymaps_pfpn/
├─ test/
│  ├─ image_canvas/
│  ├─ saliencymaps_basnet/
│  ├─ saliencymaps_pfpn/
├─ train_csv_9973.csv
```

### Usage
- Testing
1. Revise line 13 and 70 to import and load the model
2. Run ```sh test_n_eval.sh```

## Citation
If our work is helpful for your research, please cite our paper:
```
@inproceedings{Hsu-2023-posterlayout,
    title={PosterLayout: A New Benchmark and  Approach for Content-Aware Visual-Textual Presentation Layout},
    author={HsiaoYuan Hsu, Xiangteng He, Yuxin Peng, Hao Kong and Qing Zhang},
    booktitle={Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
}
```

## Contact us
For any questions or further information, please email Ms. Hsu (kslh99@stu.pku.edu.cn).