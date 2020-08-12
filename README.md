# BCRF - Detectron

bcrf-detectron is an integration of the bcrf model on top of the panoptic-fpn model of Facebook AI 
[Detectron2](https://github.com/facebookresearch/detectron2). The detectron2 repo was cloned and our bcrf model was added to it for end to end training of the bcrf model.

<div align="center">
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>



## Installation

See [INSTALL.md](INSTALL.md).

## Quick Start

See [GETTING_STARTED.md](GETTING_STARTED.md), for training, evaluation and inference

## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing Detectron

We use Detectron2 in our research to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md). 
```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```

If you are citing us, please use: 

```BibTeX
@misc{jayasumana2019bipartite,
    title={Bipartite Conditional Random Fields for Panoptic Segmentation},
    author={Sadeep Jayasumana and Kanchana Ranasinghe and Mayuka Jayawardhana and Sahan Liyanaarachchi and Harsha Ranasinghe and Sina Samangooei},
    year={2019},
    eprint={1912.05307},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
