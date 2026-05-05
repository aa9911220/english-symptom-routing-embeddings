
# English Symptom Routing Linear Probe

## Model Description

This model is an embedding-based text classifier for routing short English symptom descriptions into broad public health information categories. It was created for an Information Retrieval assignment to test whether text embeddings can support query routing before retrieval.

The model predicts one of six broad categories:

- `respiratory`
- `gastrointestinal`
- `skin`
- `neurological`
- `musculoskeletal`
- `mental_health_sleep`

The model is not a medical diagnosis system. The labels are broad information-routing categories, not diseases or clinical conditions.

**Important safety notice:**  
This model is for teaching information retrieval and text classification. It does not provide medical diagnosis, treatment advice, or emergency guidance.

## Intended Use

This model is intended for:

- teaching embedding-based text classification
- routing short symptom descriptions toward broad public health information categories
- demonstrating information retrieval query routing
- comparing lightweight classifiers trained on frozen embeddings

Example use case:

```text
Input: I have a dry cough and sore throat.
Output: respiratory
```

The predicted category could be used to select relevant public health information resources, such as respiratory health guidance.

## Out-of-Scope Use

This model must not be used for:

- medical diagnosis
- treatment recommendation
- medication advice
- emergency triage
- clinical decision-making
- replacing professional medical review
- making decisions about real patients

If someone has severe, worsening, or urgent symptoms, they should contact a qualified medical professional or emergency service.

## Research Question

Can text embeddings classify short symptom descriptions into broad health-information categories for routing users toward relevant public health resources?

## Training Data

The model was trained on a custom English symptom routing dataset.

Dataset file:

```text
symptom_routing_expanded_dataset.csv
```

Dataset size:

```text
90 examples
```

Each row contains:

| Column | Description |
|---|---|
| `text` | A short English symptom description |
| `label` | A broad public health information category |

The dataset contains six balanced categories:

| Label | Description |
|---|---|
| `respiratory` | cough, sore throat, breathing problems |
| `gastrointestinal` | stomach pain, nausea, diarrhea |
| `skin` | rash, itching, swelling |
| `neurological` | headache, dizziness, numbness |
| `musculoskeletal` | back pain, joint pain, muscle pain |
| `mental_health_sleep` | anxiety, insomnia, low mood, sleep problems |

Hugging Face Dataset:

```text
 https://huggingface.co/datasets/charlie0831/english-symptom-routing
```

## Method

The model uses a frozen embedding model to convert symptom descriptions into vector representations. A downstream classifier was then trained on these embedding vectors.

Pipeline:

```text
symptom text -> embedding model -> embedding vector -> classifier -> predicted category
```

Embedding model:

```text
nicher92/saga-embed_v1
```

Classifier:

```text
Regularized Logistic Regression
```

The embedding model was used as a frozen text encoder. Only the downstream classifier was trained for this assignment.

## Model Files

This repository contains:

| File | Description |
|---|---|
| `model_phase2.joblib` | Trained logistic regression classifier |
| `metrics.json` | Evaluation metrics |
| `predictions_phase2.csv` | Test set predictions |
| `README.md` | Model card |

## Evaluation

The model was evaluated on a held-out test set.

Data split:

```text
Test size: 0.25
Random seed: 712
```

Results:

| Metric | Score |
|---|---:|
| Accuracy | 0.957 |
| Macro F1 | 0.952 |

Macro F1 is important because it measures whether the classifier performs consistently across all categories, rather than only performing well on the most common category.

Detailed evaluation results are available in:

```text
metrics.json
predictions_phase2.csv
```

## Example Predictions

Example 1:

```text
Input: I have a fever, dry cough, and sore throat.
Predicted category: respiratory
```

Example 2:

```text
Input: My stomach hurts after eating and I feel nauseous.
Predicted category: gastrointestinal
```

Example 3:

```text
Input: I cannot sleep and I feel anxious most nights.
Predicted category: mental_health_sleep
```

## Limitations

This model was trained on a small educational dataset with 90 manually created examples. It may not generalize well to real-world symptom descriptions, different writing styles, misspellings, slang, or complex multi-symptom cases.

Some symptom descriptions can reasonably belong to more than one category. For example, a sentence mentioning both dizziness and sleep problems may be difficult to classify because it contains signals for both `neurological` and `mental_health_sleep`.

The model should only be interpreted as a broad routing tool for information retrieval experiments. It should not be interpreted as a clinical or diagnostic system.

## Demo

A working Hugging Face Space demo is available here:

```text
 https://huggingface.co/spaces/charlie0831/english-symptom-routing-demo
```

The demo lets users enter a short symptom description and returns the predicted routing category.

## Repository Links

| Resource | Link |
|---|---|
| GitHub Repository | https://github.com/aa9911220/english-symptom-routing-embeddings/tree/main |
| Hugging Face Dataset | https://huggingface.co/datasets/charlie0831/english-symptom-routing |
| Hugging Face Demo Space | https://huggingface.co/spaces/charlie0831/english-symptom-routing-demo |

## AI Tool Use

AI coding tools were used to support coding, documentation, dataset formatting, and report drafting. The outputs were manually checked and edited, especially the medical safety statements, label definitions, evaluation results, and limitations.

Using AI tools helped speed up implementation, but the project still required manual understanding of the data, embedding pipeline, classifier training, and evaluation results.

## License

This model is released under the MIT License.

