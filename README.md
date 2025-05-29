# Paths-over-Graph (PoG) Code Documentation

## News!
Our paper has been accepted for publication at WWW 2025! 

## How to cite
If you are interested or inspired by this work, you can cite us by:
```sh
@inproceedings{tan2025paths_over_graph,
  title={Paths-over-graph: Knowledge graph empowered large language model reasoning},
  author={Tan, Xingyu and Wang, Xiaoyang and Liu, Qing and Xu, Xiwei and Yuan, Xin and Zhang, Wenjie},
  booktitle={Proceedings of the ACM on Web Conference 2025},
  pages={3505--3522},
  year={2025}
}
```

## Get started
Before running PoG, please ensure you have successfully installed **Freebase** on your local machine. The comprehensive installation instructions and necessary configuration details can be found in the `/Freebase/README.md`.

The required libraries for running ToG can be found in `requirements.txt`. You must use your own API in the run_LLM function of utils.py for the APIs.

To set up the environment, install the required dependencies using:

```bash
cd PoG
```

```bash
pip install -r requirements.txt
```


## Running the Dataset for Single-Keyword Questions

Use the following command to run the dataset for single-keyword questions:

```bash
python PoG_single.py <Dataset_name> <sum/unsum> <beam_search:1, 12, 13, 123> <PoG/PoGE> <gpt3/gpt4> <max_depth 1/2/3>
```

### Arguments:
- `<Dataset_name>`: The dataset to use. Options:
  - `webqsp`
  - `cwq`
  - `grailqa`
  - `webquestions`
  - `simpleqa`
- `<sum/unsum>`: 
  - `sum` for using path summary
  - `unsum` for not using summary
- `<beam_search>`: Beam search strategy:
  - `1` for only using fuzzy selection
  - `12` for step1 + BranchReduced
  - `13` for step1 + precise selection
  - `123` for using all steps
- `<PoG/PoGE>`:
  - `PoG` for using all relations
  - `PoGE` for using a random single relation
- `<gpt3/gpt4>`:
  - `gpt3` for using GPT-3
  - `gpt4` for using GPT-4
- `<max_depth>`: Maximum search depth:
  - `1` for using only 1-hop
  - `2` for using 2-hop
  - `3` for using 3-hop

### KG usage:
PoG utilze the Freebase KG. For more details about Freebase installation, please refer to the Freebase folder.

## Running the Dataset for Multi-Keyword Questions

Use the following command to run the dataset for multi-keyword questions:

```bash
python PoG_multi.py <Dataset_name> <sum/unsum> <beam_search:1, 12, 13, 123> <PoG/PoGE> <gpt3/gpt4> <max_depth 1/2/3>
```

### Arguments:
The arguments are the same as those for `PoG_single.py`.

### Subgraph Loading:
PoG_multi will load the subgraph at maximum depths first from KG as the database preparation. The loading time depends on the environment setup and memory allocated for the freebase server.

### KG usage:
PoG utilze the Freebase KG. For more details about Freebase installation, please refer to the Freebase folder.
## Checking the Answer

Use the following command to check the answer:

```bash
python check_answer.py <Dataset_name> <sum/unsum> <beam_search:1, 12, 13, 123> <PoG/PoGE> <gpt3/gpt4> <max_depth 1/2/3>
```

### Arguments:
The arguments are the same as those for `PoG_single.py`.

## Notes
- Ensure that the dataset files and model configurations are correctly set up before running the scripts.
- Use appropriate depth values based on the complexity of the dataset and required accuracy.
- Experiment with different beam search strategies to find the optimal balance between speed and accuracy.

For any issues, please refer to the error messages or modify the script parameters accordingly.

## Claims
This project uses the Apache 2.0 protocol. The project assumes no legal responsibility for any of the model's output and will not be held liable for any damages that may result from the use of the resources and output.
