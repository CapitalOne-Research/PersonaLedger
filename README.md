<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> PersonaLedger: Generating Realistic Financial Transactions with Persona Conditioned LLMs and Rule Grounded Feedback  </h1>

<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://dhyuan99.github.io/" target="_blank" style="text-decoration: none;">Dehao Yuan</a><sup>1</sup>&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=emHB_OEAAAAJ&hl=en" target="_blank" style="text-decoration: none;">Tyler Farnan</a><sup>1</sup>&nbsp;&nbsp;
    <a href="https://www.stefantesliuc.com/about" target="_blank" style="text-decoration: none;">Stefan Tesliuc</a><sup>1</sup>&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=FeCagRUAAAAJ&hl=en" target="_blank" style="text-decoration: none;">Doron L Bergman</a><sup>1</sup>&nbsp;&nbsp;
    <a href="https://yulun-rayn.github.io/" target="_blank" style="text-decoration: none;">Yulun Wu</a><sup>1</sup>&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=EcHTiyIAAAAJ&hl=en" target="_blank" style="text-decoration: none;">Xiaoyu Liu</a><sup>2</sup>&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=UKAsIsUAAAAJ&hl=en" target="_blank" style="text-decoration: none;">Minghui Liu</a><sup>3</sup>&nbsp;&nbsp;
    <a href="http://jamesmontgomery.us/" target="_blank" style="text-decoration: none;">James Montgomery</a><sup>1</sup>&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=zzBcUpEAAAAJ&hl=en" target="_blank" style="text-decoration: none;">Nam H Nguyen</a><sup>1</sup>&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=ClqvGRQAAAAJ&hl=en" target="_blank" style="text-decoration: none;">C. Bayan Bruss</a><sup>1</sup>&nbsp;&nbsp;
    <a href="https://furong-huang.com/" target="_blank" style="text-decoration: none;">Furong Huang</a><sup>3</sup>
    <br><br>
    <sup>1</sup>Capital One Research, &nbsp;&nbsp;
    <sup>2</sup>Google Inc. &nbsp;&nbsp;<br>
    <sup>3</sup>University of Maryland, College Park,&nbsp;&nbsp;
    <br><br>
    <a href="https://arxiv.org/abs/2601.03149" target="_blank" style="text-decoration: none;">[Paper]</a> &nbsp;&nbsp
    <a href="https://huggingface.co/datasets/capitalone/PersonaLedger" target="_blank" style="text-decoration: none;">[Dataset]</a> &nbsp;&nbsp
    <a href="https://github.com/CapitalOne-Research/PersonaLedger" target="_blank" style="text-decoration: none;">[Code]</a> &nbsp;&nbsp
    <br>
</p>

**For code:** [THIS](./generation) generates PersonaLedger; [THIS](./evaluation) evaluates 14 models on PersonaLedger.

<div align="center">
  <img src="./image.jpg" width="100%" alt="PersonaLedger Architecture">
</div>

### What does PersonaLedger feature?
* a synthetic dataset of 30 million financial transaction.
* persona-driven LLMs for **behavioral diversity**.
* programmatic engine to enforce **accounting correctness**.

### Benchmarking Tasks
1. **Insolvency Prediction** (Referred to as "Illiquidity Classification" in the paper)
    * Task Type: Sequence Classification (User-Level).
    * Goal: Given n-month transaction history, predict whether the user will become "illiquid" in the near future.
    * Definition of Insolvency/Illiquidity: A state where a user’s credit card balance exceeds their available cash flow and credit limit at specific moments.
2. **Identity Theft** (Referred to as "Identity Theft Segmentation" in the paper)
    * Task Type: Segmentation (Event-Level).
    * Goal: Identify fraudulent transactions hidden within a user's legitimate history.
    * Data Composition: The dataset injects one day of chronologically aligned transactions from a secondary user (the thief) into the primary user's history.

### Dataset Structure
```
├── README.md
├── identity_theft_1months/          # Identity theft detection data (1-month period)
│   └── train.parquet
│   └── test.parquet                 # parquet files contain labels.
├── identity_theft_3months/          # Identity theft detection data (3-month period)
│   └── train.parquet
│   └── test.parquet                 # parquet files contain labels.
├── insolvency_prediction_1months/   # Insolvency prediction data (1-month period)
│   └── train.parquet
│   └── test.parquet
│   └── labels.json
├── insolvency_prediction_3months/   # Insolvency prediction data (3-month period)
│   └── train.parquet
│   └── test.parquet
│   └── labels.json
└── raw_generation/                  # Raw generated data
    ├── insolvent/                   # Insolvent entity samples
    └── normal/                      # Normal entity samples
```

### Citation
```
@misc{yuan2026personaledgergeneratingrealisticfinancial,
      title={PersonaLedger: Generating Realistic Financial Transactions with Persona Conditioned LLMs and Rule Grounded Feedback}, 
      author={Dehao Yuan and Tyler Farnan and Stefan Tesliuc and Doron L Bergman and Yulun Wu and Xiaoyu Liu and Minghui Liu and James Montgomery and Nam H Nguyen and C. Bayan Bruss and Furong Huang},
      year={2026},
      eprint={2601.03149},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.03149}, 
}
```
