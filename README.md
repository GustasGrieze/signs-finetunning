# Lithuanian Road Sign Captioning with Qwen3-VL

This project fine-tunes a vision-language model to generate short Lithuanian captions for road sign images.
The goal was to improve the model's ability to describe visible road signs, their meaning, readable text, numbers, arrows, and uncertainty when signs are unclear.

The project compares the original baseline model with a fine-tuned LoRA adapter trained on a custom Lithuanian road sign captioning dataset.

## Project Overview

The selected base model was:

```text
unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit
```

The model was fine-tuned using LoRA adapters with Unsloth. The final adapter was uploaded to Hugging Face:

```text
https://huggingface.co/RidaKo/qwen3-vl-road-signs-lt-lora
```

The task was formulated as image-conditioned text generation. For each input image, the model received a Lithuanian instruction prompt and generated a short Lithuanian caption focused on road signs.

## Final Prompt

The same prompt was used for both baseline and fine-tuned inference:

```text
Trumpai, 1–3 sakiniais, aprašyk nuotraukoje matomus kelio ženklus lietuviškai. Paminėk ženklo tipą, reikšmę, matomą tekstą, skaičius ar rodykles. Jei ženklas neaiškus, per toli arba tekstas neįskaitomas, taip ir parašyk. Nespėliok nematomų detalių.
```

## Repository Contents

```text
.
├── Qwen3_VL_8B_road_signs_.ipynb       # Final training and evaluation notebook
├── README.md                           # Project description
├── report/                             # Final report files, if included
├── results/                            # Evaluation outputs, if included
└── data/                               # Dataset files or references, if included
```

The main notebook contains:

* dataset loading;
* train / validation / test split handling;
* baseline inference;
* LoRA fine-tuning;
* fine-tuned inference;
* automatic metric calculation;
* manual scoring export;
* result comparison.

## Dataset

The dataset contains Lithuanian road sign images with human-written Lithuanian captions.

Final dataset split:

| Split      | Images |
| ---------- | -----: |
| Training   |    161 |
| Validation |     18 |
| Test       |     50 |

The test set was held out and was not used during training.

The images were resized so that the longest side was at most 768 pixels while preserving the original aspect ratio.

## Training Setup

| Parameter                   |                                Value |
| --------------------------- | -----------------------------------: |
| Base model                  |           Qwen3-VL 8B Instruct 4-bit |
| Fine-tuning method          |                                 LoRA |
| LoRA rank                   |                                   32 |
| LoRA alpha                  |                                   32 |
| LoRA dropout                |                                 0.05 |
| Epochs                      |                                    4 |
| Learning rate               |                                 5e-5 |
| Batch size per device       |                                    1 |
| Gradient accumulation steps |                                    4 |
| Effective batch size        |                                    4 |
| Max generated tokens        |                                  256 |
| Trainable parameters        |          102,693,888 / 8,869,817,584 |
| GPU                         | NVIDIA GeForce RTX 5070 Ti, 15.92 GB |

Only approximately 1.16% of the model parameters were updated during LoRA fine-tuning.

## Evaluation

The baseline and fine-tuned models were evaluated on the same fixed 50-image test set.

The evaluation included:

* BLEU;
* chrF;
* ROUGE-L;
* road sign keyword precision / recall / F1;
* visible text and number precision / recall / F1;
* manual scoring from 0 to 10 using a road-sign-specific rubric.

## Automatic Results

| Model      |   BLEU |   chrF | ROUGE-L | Keyword F1 | Visible Value F1 |
| ---------- | -----: | -----: | ------: | ---------: | ---------------: |
| Baseline   |  0.339 | 28.810 |   0.130 |      0.097 |            0.131 |
| Fine-tuned | 10.669 | 45.230 |   0.318 |      0.538 |            0.438 |

The fine-tuned model improved across all main automatic metrics.

## Manual Scoring Results

Manual scoring was performed using five criteria:

* sign presence;
* sign type / meaning;
* readable text / numbers;
* uncertainty handling;
* Lithuanian description quality.

Each criterion was scored from 0 to 2 points, for a maximum total score of 10.

| Criterion                 | Baseline | Fine-tuned |
| ------------------------- | -------: | ---------: |
| Sign presence             |     1.18 |       1.54 |
| Sign type / meaning       |     0.68 |       1.12 |
| Readable text / numbers   |     0.54 |       0.92 |
| Uncertainty handling      |     0.32 |       0.94 |
| Lithuanian quality        |     0.62 |       1.48 |
| **Total score out of 10** | **3.34** |   **6.00** |

Manual scoring confirmed the trend shown by the automatic metrics: the fine-tuned model produced more useful and more natural Lithuanian captions.

## Supplementary Materials

Automatic evaluation outputs and generated captions:

```text
https://docs.google.com/spreadsheets/d/1m1NXrT2rb1gr1VzlDoldZHTy0sU_G8nncXiD-w5xo3Q/edit?usp=sharing
```

Manual scoring spreadsheet:

```text
https://docs.google.com/spreadsheets/d/1YTpa-ECTzPHJ2lDMHfK3sxF6wN2-GQjU14fn1TodYt0/edit?usp=sharing
```

Fine-tuned LoRA adapter:

```text
https://huggingface.co/RidaKo/qwen3-vl-road-signs-lt-lora
```

Base model:

```text
https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit
```

## How to Use the Adapter

The uploaded Hugging Face repository contains a LoRA adapter, not a standalone full model. It should be loaded together with the base Qwen3-VL model.

Example loading code:

```python
from unsloth import FastVisionModel

base_model = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
adapter_model = "RidaKo/qwen3-vl-road-signs-lt-lora"

model, tokenizer = FastVisionModel.from_pretrained(
    base_model,
    load_in_4bit=True,
)

model.load_adapter(adapter_model)
```

## Limitations

The fine-tuned model performs better than the baseline, but it is not perfect.

Known limitations:

* it can confuse visually similar signs;
* it can miss small or distant signs;
* it can hallucinate text or numbers;
* it can describe unclear signs too confidently;
* it should not be used as a safety-critical traffic sign recognition system.

The model should be treated as an experimental Lithuanian road sign captioning adapter.
