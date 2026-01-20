# PersonaLedger Generation Engine

This folder contains the implementation of PersonaLedger generation engine, which includes two parts:
1. **Augmented personas generation**: Starting from Nvidia-Nemotron-Persona, use an Llama-3.3-70B to inference credit limit, etc. for the subsequent transactions generation.
2. **persona-conditioned grounded transaction sequences generation.** Using the generated personas in Step 1, we use Llama-3.3-70B, coupled with a programmatic engine to generate logically grounded and feasible transactions.
3. **Benchmarking tasks creation.** After generating many transactions sequences in Step 2, we postprocess the sequences and curate two benchmarking tasks, *illiquidity classification* and *identity theft classification*. The created benchmarking data can be directly used in [evaluation](../evaluation).

## Environment Setup and LLM Download
Our generation engine utilizes `vllm` to inference, due to its excellent efficiency. We use [`Llama-3.3-70B-Instruct`](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct). To setup the environment,
```
conda create -n vllm python=3.12 -y
conda activate vllm
pip install -r requirements.txt
```
Then download the model weights from [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) and put it at your convenient place. Rename the folder as `70B`. Then use `vllm` to start a server:
```
vllm serve ./70B --tensor_parallel_size 8
```
Ideally, the server will be on `http://localhost:8000/v1`, and during inference we will keep sending requests to this local host.

#### LLM API (Optional)
It is totally feasible to use closed-source LLM API to do the generation, which can further the realism of the data. To achieve this, simply implement your own lm_provider at `./utils/lm_providers.py` and replace the `LocalServer` with your implementation. **Warning:** The API cost can be prohibitively expensive. For example, generating one user can cost 10 USD, and generating 25,000 users will cost 250,000 USD.

## Step 1: Augmented Personas Generation
**Pre-computed augmented personas.** We have already included the augmented personas (100,000 users) we used in the paper at [`./personas/augmented_persona`](./personas/augmented_persona). If you just want to use it rather than generating by yourself, you can find them there.

**Reproducing the generation.** If you want to adjust the generation process by yourself, here is the instruction. We input the [Nvidia-Nemotron-Persona](https://huggingface.co/datasets/nvidia/Nemotron-Personas) and use LLM to infer the financial status. The implementation is in [`./Generators/generator_more_info.py`](./Generators/generator_more_info.py). We include the Nemotron-Persona dataset in this repository [`personas/nemotron_personas`](`personas/nemotron_personas`), so there is no need to download the dataset. To run the inference on a single batch of 1000 users:
```
python main.py experiment=generate_more_info experiment.file_idx=000000
```
Or to run the inference on batch, which utilizes `vllm`'s continuous batching mechanism, simple run this
```
bash ./scripts/generate_financial_info.sh
```

It will save the augmented personas to [`./personas/augmented_persona`](./personas/augmented_persona), overwriting our pre-computed ones.

## Step 2: Transaction Sequences Generation
**Pre-computed transactions.** To have a glance of how the outputs look like, see [`./generated_transactions/000000`](generated_transactions/000000). For each user, it contains two files:
1. `./generated_transactions/000000/000000.json`
2. `./generated_transactions/000000/conversation_000000.json`)

The first file contains the user personas, financial info, and the generated transactions. The second file contains the entire tracking of the generation, including running balance, user daily trajectory, etc.. This can be useful for tracking the mechanism behind the transactions generation.

**Reproducing the generation.** If you want to reproduce or adjust the generation process by yourself, here is an example script to run:
```
python main.py experiment=generate experiment.json_name="000000" experiment.start_idx=0 experiment.end_idx=250
```
This is generating transactions for batch-000000 (containing 1000 users), within each batch generating from user 0 to user 250. This will generate transactions of 250 users in parallel and therefore utilize `vllm`'s continuous batching mechanism. 

To see all the scripts to run, check [`./scripts/generate.sh`](./scripts/generate.sh).

**Tuning Human-Implemented Rules.** If you want to add more rules and constraints to the generation, as mentioned in the paper, check [`./Generators/generation_tracker.py`](./Generators/generation_tracker.py). You will need to
1. Implement your own rules following the `Rule` template.
2. Add your rules to the `GenerationTracker::applied_rules` which looks like
```
self.applied_rules = [
    Holiday           (None),
    AmountPolarity    (None),
    Subscriptions     (user_financial_profile),
    PaymentAndInterest(user_financial_profile),
    LastTimeGas       (user_financial_profile),
    LastTimeGrocery   (user_financial_profile),
    RandomEvents      (user_financial_profile, self.cfg.experiment.random_event_p),
    DebitCard         (user_financial_profile),
    CreditBalance     (user_financial_profile),
    CreditAvailability(user_financial_profile) # DebitCard, CreditBalance and CreditAvailability must be placed at last.
]
```

## Step 3: Postprocess Data
After obtaining the generated transactions, we convert them from json to parquet to optimize storage. This is done by
```
python main.py experiment=postprocess
```
This will convert the generated transactions in `./generated_transactions` to `./generated_transactions/raw_generation`.

## Step 4: Task Creation
Finally, create the two benchmarking tasks. It will load from `./generated_transactions/raw_generation` and create train-test split saved at `./generated_transactions/raw_generation/identity_theft_3months` and `./generated_transactions/raw_generation/illiquidity_prediction_3months`.
```
python main.py experiment=task_creation experiment.what_task=identity_theft experiment.month_window=3
python main.py experiment=task_creation experiment.what_task=illiquidity_classification experiment.month_window=3
```
