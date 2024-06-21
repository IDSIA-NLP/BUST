# BUST
BUST is a comprehensive benchmark for evaluating synthetic text detectors, focusing on their effectiveness against outputs from various Large Language Models (LLMs). BUST evaluates detectors using a wide range of metrics including linguistic features, readability, and writer attitudes, aiming to identify spurious signals that may influence detection. The benchmark not only ranks detectors but also analyzes their performance correlations with specific metrics and provides insights into biases and robustness against different LLM outputs. BUST is designed as a dynamic resource, continuously updated to stay relevant in the rapidly evolving field of LLM-generated content detection.

## Benchmark and Dataset
Visit the [BUST Benchmark Page](https://bust.nlp.idsia.ch) to download the test dataset and upload predictions of detectors to test new detectors against the benchmark.


## Setup

1. **Python Version**: Ensure you have Python 3.10 installed.
2. **Install Requirements**:
    ```sh
    pip install -r requirements.txt
    ```
   - For running the LLMDet detector:
     ```sh
     pip install -r requirements_llmdet.txt
     ```
   - For running the Ghostbuster detector:
     ```sh
     pip install -r requirements_ghostbuster.txt
     ```

3. **Environment Variables**: Create a `.env` file with the following keys:
    ```plaintext
    # OpenAI keys
    OPENAI_KEY=your_openai_key
    OPENAI_ORG=your_openai_org

    # Replicate key
    REPLICATE_API_TOKEN=your_replicate_api_token

    # Huggingface key
    HF_API_TOKEN=your_huggingface_api_token

    # GPTZero key
    GPT_ZERO_API_KEY=your_gpt_zero_api_key
    ```

## Pipeline

Runs files in the following order:

1. **Generate Synthetic Text Dataset**:
    - Set flags in `configs/configs.yml` to run the ablation studies.
    - Configure models in the `src/create_generated_text_dataset.py` file.

2. **Merge Datasets**:
    If you have scattered data from several runs of `src/create_generated_text_dataset.py` that need to be merged into a single data file.

3. **Reformat Dataset**:
    - Reformats data so that all models' outputs and human responses are in their own rows.

4. **Create Detector Dataset**:
    - Create detector predictions and statistics.
    - Set the metrics to be used in the file.

5. **Detect with LLMDet**:
    - Uses the LLMDet detector.
    - Requires a different conda environment.

6. **Ghostbuster Detector**:
    - Runs the Ghostbuster detector.
    - Requires a different conda environment.


## Folder Structure
```
BUST/
├── config_pipeline.yaml
├── configs/
│   └── configs.yml
├── data/                  # Ignored from git
├── .env
├── .gitignore
├── logs/                  # Ignored from git
├── notebooks/
│   └── merge_datasets.ipynb
├── out/                   # Ignored from git
├── README.md
├── requirements_ghostbuster.txt
├── requirements_llmdet.txt
├── requirements.txt
├── results/               # Ignored from git
└── src/
    ├── create_detector_dataset.py
    ├── create_generated_text_dataset.py
    ├── detect_with_gpt_zero.py
    ├── detect_with_llm_det.py
    ├── detect_with_radar_vicunia.py
    ├── generators.py
    ├── hfpipeline_classification.py
    ├── __init__.py
    ├── reformat_dataset.py
    ├── step_compute_writers_attitude.py
    └── utils.py
```

### Ignored Directories
- `data/`: Contains dataset files.
- `logs/`: Stores log files.
- `out/`: Outputs contains results form meta analysis.
- `results/`: Contains resulting benchmark files of detector evaluations.

## Citation

If you use BUST in your research, please cite the following paper:

```plaintext
@inproceedings{cornelius-etal-2024-bust,
    title = "{BUST}: Benchmark for the evaluation of detectors of {LLM}-Generated Text",
    author = "Cornelius, Joseph  and
      Lithgow-Serrano, Oscar  and
      Mitrovic, Sandra  and
      Dolamic, Ljiljana  and
      Rinaldi, Fabio",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.444",
    pages = "8022--8050",
    abstract = "We introduce BUST, a comprehensive benchmark designed to evaluate detectors of texts generated by instruction-tuned large language models (LLMs). Unlike previous benchmarks, our focus lies on evaluating the performance of detector systems, acknowledging the inevitable influence of the underlying tasks and different LLM generators. Our benchmark dataset consists of 25K texts from humans and 7 LLMs responding to instructions across 10 tasks from 3 diverse sources. Using the benchmark, we evaluated 5 detectors and found substantial performance variance across tasks. A meta-analysis of the dataset characteristics was conducted to guide the examination of detector performance. The dataset was analyzed using diverse metrics assessing linguistic features like fluency and coherence, readability scores, and writer attitudes, such as emotions, convincingness, and persuasiveness. Features impacting detector performance were investigated with surrogate models, revealing emotional content in texts enhanced some detectors, yet the most effective detector demonstrated consistent performance, irrespective of writer{'}s attitudes and text styles. Our approach focused on investigating relationships between the detectors{'} performance and two key factors: text characteristics and LLM generators. We believe BUST will provide valuable insights into selecting detectors tailored to specific text styles and tasks and facilitate a more practical and in-depth investigation of detection systems for LLM-generated text.",
}
```
