# A tutorial on neural theorem proving

*Neural theorem proving* combines neural language models with formal proof assistants.\
This tutorial introduces two research threads in neural theorem proving via interactive Jupyter notebooks.


## Part I : Next-step suggestion

Builds a neural next-step suggestion tool, introducing concepts and past work in neural theorem proving along the way.

<img src="./partI_nextstep/notebooks/images/llmsuggest/llmsuggest.gif" width="350"/>

#### Notebooks:
| Topic | Notebook | 
|:-----------------------|-------:|
| 0. Intro            | [notebook](./partI_nextstep/notebooks/I_nextstep_lean__part0_intro.ipynb) |
| 1. Data             | [notebook](./partI_nextstep/notebooks/I_nextstep_lean__part1_data.ipynb) |
| 2. Learning         | [notebook](./partI_nextstep/notebooks/I_nextstep_lean__part2_learn.ipynb) |
| 3. Proof Search     | [notebook](./partI_nextstep/notebooks/I_nextstep_lean__part3_proofsearch.ipynb) |
| 4. Evaluation       | [notebook](./partI_nextstep/notebooks/I_nextstep_lean__part4_evaluation.ipynb) |
| 5. `llmsuggest`        | [notebook](./partI_nextstep/notebooks/I_nextstep_lean__part5_llmsuggest.ipynb) |

All notebooks are in ([`partI_nextstep/notebooks`](./partI_nextstep/notebooks)). See [`partI_nextstep/ntp_python`](./partI_nextstep/ntp_python) and [`partI_nextstep/ntp_lean`](./partI_nextstep/ntp_lean) for the Python and Lean files covered in the notebooks.

#### Setup:
Please follow the setup instructions in [`partI_nextstep/README.md`](./partI_nextstep/README.md).

## Part II : Language cascades
Chain together language models to guide formal proof search with informal proofs.


#### Notebooks:
| Topic | Notebook | 
|:-----------------------|-------:|
| 1. Language model cascades | [notebook](./partII_dsp/notebooks/II_dsp__part1_intro.ipynb) |
| 2. Draft, Sketch, Prove | [notebook](./partII_dsp/notebooks/II_dsp__part2_dsp.ipynb) |

All notebooks are in ([`partII_dsp/notebooks`](./partII_dsp/notebooks)).

#### Setup:
Please follow the setup instructions in [`partII_dsp/README.md`](./partII_dsp/README.md).


-------
### History
These materials were originally developed as part of a IJCAI 2023 tutorial. \
Slides for the 1 hour summary presentation given at IJCAI 2023 are [here](https://wellecks.com/data/welleck2023ntp_tutorial.pdf). 

#### Citation

If you find this tutorial or repository useful in your work, please cite:
```
@misc{ntptutorial,
  author = {Sean Welleck},
  title = {Neural theorem proving tutorial},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/wellecks/ntptutorial}},
}
```
