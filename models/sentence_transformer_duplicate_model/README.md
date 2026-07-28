---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:258743
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: 'CAT Preparation: Is it that valuable doing MBA from IITs as compared
    to other reputed management colleges?'
  sentences:
  - What does it feel like to be a famous writer?
  - What is the cheapest way to send money to India from the Korea?
  - I have call from IIT Delhi for MBA, should I opt for it or prepare for CAT next
    year?
- source_sentence: What is it like working in a call centre?
  sentences:
  - What are the strongest majors in terms of job prospects and what are the weakest
    majors at Park University?
  - How can I read someone else's WhatsApp messages without using his/her phone?
  - What is it like to work at a call centre?
- source_sentence: What are all of the American Pie movies, in order?
  sentences:
  - What are some tips on making it through the job interview process at Chart Industries?
  - Why should one watch American Pie Series once in his lifetime?
  - Why does India oppose CPEC?
- source_sentence: How can one improve patience?
  sentences:
  - What is meant by "without avail" or "to no avail"?
  - What is the best way to practice patience?
  - What is the procedure to get a distributorship from any brand (like Nike or Adidas)
    to sell their products online like on Snapdeal and Paytm?
- source_sentence: Who are the Top Question Writers for 2016?
  sentences:
  - How much time does TCS take to give the off campus freshers the joining letter?
  - Who are the Top Question Writers of 2016?
  - How will Donald Trump getting elected as the President of the United States affect
    the relations of the USA and India?
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Who are the Top Question Writers for 2016?',
    'Who are the Top Question Writers of 2016?',
    'How will Donald Trump getting elected as the President of the United States affect the relations of the USA and India?',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9856, 0.1674],
#         [0.9856, 1.0000, 0.1873],
#         [0.1674, 0.1873, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 258,743 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                        | sentence_1                                                                        | label                                                          |
  |:---------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type     | string                                                                            | string                                                                            | float                                                          |
  | modality | text                                                                              | text                                                                              |                                                                |
  | details  | <ul><li>min: 7 tokens</li><li>mean: 15.36 tokens</li><li>max: 40 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 15.07 tokens</li><li>max: 47 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.38</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                            | sentence_1                                                               | label            |
  |:----------------------------------------------------------------------|:-------------------------------------------------------------------------|:-----------------|
  | <code>Which is the best food for Bernese Mountain Dog puppies?</code> | <code>What is the best food for a 5 week old puppy?</code>               | <code>1.0</code> |
  | <code>Which restaurants accept Sodexo coupons in Mumbai?</code>       | <code>Which restaurants accepts sodexo coupons in pune?</code>           | <code>0.0</code> |
  | <code>When will corruption in India end?</code>                       | <code>What steps must Indians take to make India corruption free?</code> | <code>0.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss",
      "cos_score_transformation": "torch.nn.modules.linear.Identity"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 2
- `disable_tqdm`: True
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 2
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: True
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.0309 | 500   | 0.2169        |
| 0.0618 | 1000  | 0.1707        |
| 0.0928 | 1500  | 0.1464        |
| 0.1237 | 2000  | 0.1424        |
| 0.1546 | 2500  | 0.1390        |
| 0.1855 | 3000  | 0.1391        |
| 0.2164 | 3500  | 0.1360        |
| 0.2473 | 4000  | 0.1310        |
| 0.2783 | 4500  | 0.1280        |
| 0.3092 | 5000  | 0.1284        |
| 0.3401 | 5500  | 0.1241        |
| 0.3710 | 6000  | 0.1224        |
| 0.4019 | 6500  | 0.1212        |
| 0.4328 | 7000  | 0.1202        |
| 0.4638 | 7500  | 0.1224        |
| 0.4947 | 8000  | 0.1181        |
| 0.5256 | 8500  | 0.1175        |
| 0.5565 | 9000  | 0.1191        |
| 0.5874 | 9500  | 0.1191        |
| 0.6184 | 10000 | 0.1205        |
| 0.6493 | 10500 | 0.1152        |
| 0.6802 | 11000 | 0.1190        |
| 0.7111 | 11500 | 0.1174        |
| 0.7420 | 12000 | 0.1174        |
| 0.7729 | 12500 | 0.1132        |
| 0.8039 | 13000 | 0.1143        |
| 0.8348 | 13500 | 0.1092        |
| 0.8657 | 14000 | 0.1123        |
| 0.8966 | 14500 | 0.1122        |
| 0.9275 | 15000 | 0.1113        |
| 0.9584 | 15500 | 0.1126        |
| 0.9894 | 16000 | 0.1069        |
| 1.0203 | 16500 | 0.1060        |
| 1.0512 | 17000 | 0.0998        |
| 1.0821 | 17500 | 0.0989        |
| 1.1130 | 18000 | 0.1004        |
| 1.1440 | 18500 | 0.1021        |
| 1.1749 | 19000 | 0.1017        |
| 1.2058 | 19500 | 0.1009        |
| 1.2367 | 20000 | 0.1033        |
| 1.2676 | 20500 | 0.0998        |
| 1.2985 | 21000 | 0.1002        |
| 1.3295 | 21500 | 0.0982        |
| 1.3604 | 22000 | 0.0967        |
| 1.3913 | 22500 | 0.0958        |
| 1.4222 | 23000 | 0.0993        |
| 1.4531 | 23500 | 0.1009        |
| 1.4840 | 24000 | 0.0985        |
| 1.5150 | 24500 | 0.0976        |
| 1.5459 | 25000 | 0.0972        |
| 1.5768 | 25500 | 0.0967        |
| 1.6077 | 26000 | 0.1005        |
| 1.6386 | 26500 | 0.0971        |
| 1.6696 | 27000 | 0.0988        |
| 1.7005 | 27500 | 0.0968        |
| 1.7314 | 28000 | 0.0990        |
| 1.7623 | 28500 | 0.0964        |
| 1.7932 | 29000 | 0.0968        |
| 1.8241 | 29500 | 0.0955        |
| 1.8551 | 30000 | 0.0978        |
| 1.8860 | 30500 | 0.0976        |
| 1.9169 | 31000 | 0.0937        |
| 1.9478 | 31500 | 0.0955        |
| 1.9787 | 32000 | 0.0992        |


### Training Time
- **Training**: 31.8 minutes

### Framework Versions
- Python: 3.12.13
- Sentence Transformers: 5.6.0
- Transformers: 5.13.1
- PyTorch: 2.11.0+cu128
- Accelerate: 1.14.0
- Datasets: 4.0.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->