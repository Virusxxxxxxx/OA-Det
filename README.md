# Shared-weight-based Multi-dimensional Feature Alignment Network for Oriented Object Detection in Remote Sensing Imagery
This project is the official implementation of OA-Det.  
The paper has been made available in open access format at [https://www.mdpi.com/1424-8220/23/1/207].  
Our model is based on the [MMRotate](https://github.com/open-mmlab/mmrotate), an open-source toolbox for oriented object detection based on PyTorch.

## Highlight

<img src="https://github.com/Virusxxxxxxx/OA-Det/blob/master/resources/ga.png?raw=true"  width="1000"/>

## Main Results

|  Method |  Dataset |  Backbone  |  Input Size |  mAP |
| ------- | -------- | ---------- | ----------- | ---- |
| OA-Det  | DOTA 1.0 | ResNet-101 | 1024 x 1024 | 78.11 |
| OA-Det  | HRSC2016 | ResNet-101 |  800 x 800  | 90.10 |
| OA-Det  | UCAS-AOD | ResNet-101 |  800 x 800  | 90.29 |

* Visualization results on the test set of DOTA.

<img src="https://github.com/Virusxxxxxxx/OA-Det/blob/master/resources/dota.png?raw=true" width="1000"/>

* Visualization results on HRSC2016.

<img src="https://github.com/Virusxxxxxxx/OA-Det/blob/master/resources/hrsc.png?raw=true" width="1000"/>

* Visualization results on UCAS-AOD.

<img src="https://github.com/Virusxxxxxxx/OA-Det/blob/master/resources/ucas.png?raw=true" width="1000"/>

## Citation
If you use our work in your research, please cite this project.
```bibtex
@article{OADet,
  title={Shared-Weight-Based Multi-Dimensional Feature Alignment Network for Oriented Object Detection in Remote Sensing Imagery},
  author={Hu, Xinxin and Zhu, Changming},
  journal={Sensors},
  volume={23},
  number={1},
  pages={207},
  year={2022},
  publisher={MDPI}
}
```

## License
This project is released under the [Apache 2.0 license](LICENSE).
