
```markdown
# Symptom Description Routing for Public Health Information Retrieval

This project trains embedding-based classifiers to route short English symptom descriptions into broad public health information categories. It was created for an Information Retrieval assignment to test whether frozen text embeddings can support query routing before retrieval.

The final dataset contains 90 examples. Four classifiers were trained and evaluated on the same train/test split. The best model achieved **0.957 accuracy** and **0.952 macro F1** on the held-out test set.

**Important safety notice:**  
This model is for teaching information retrieval and text classification. It does not provide medical diagnosis, treatment advice, or emergency guidance.

## Research Question

Can text embeddings classify short symptom descriptions into broad health-information categories for routing users toward relevant public health resources?

## Domain Issue

People often describe symptoms in short, informal text. For public health information retrieval, these descriptions need to be routed to relevant information categories before retrieval. The challenge is that symptom descriptions can be brief, ambiguous, and may contain overlapping signals from multiple categories.

This project uses embeddings to represent short symptom descriptions as semantic vectors, allowing lightweight classifiers to learn routing patterns from a small custom dataset.

## Task Definition

Given a short symptom description, the system predicts one broad health-information category. The predicted category can be used as a routing signal for public health information retrieval.

The system is designed to:

- classify short English symptom descriptions
- route user queries toward broad public health information categories
- demonstrate embedding-based text classification
- compare several classifiers trained on frozen embeddings

The system is not designed to:

- diagnose diseases
- recommend medicine
- provide treatment advice
- perform emergency triage
- replace medical professionals
- make clinical decisions

## Labels

The dataset uses six broad routing labels.

| Label | Meaning |
|---|---|
| `respiratory` | cough, sore throat, breathing problems |
| `gastrointestinal` | stomach pain, nausea, diarrhea |
| `skin` | rash, itching, swelling |
| `neurological` | headache, dizziness, numbness |
| `musculoskeletal` | back pain, joint pain, muscle pain |
| `mental_health_sleep` | anxiety, insomnia, low mood, sleep problems |

These labels are information categories, not medical diagnoses.

## Dataset

The custom dataset is stored in:

```text
symptom_routing_expanded_dataset.csv
```

The dataset contains 90 manually written English symptom descriptions across six balanced public health information categories.

Each row contains:

| Column | Description |
|---|---|
| `text` | A short English symptom description |
| `label` | The broad public health information category |

Example rows:

```csv
text,label
"I have a dry cough and sore throat, and I feel feverish.",respiratory
"My stomach hurts after eating and I feel nauseous.",gastrointestinal
"I have an itchy red rash on my arms.",skin
"I feel dizzy and have a strong headache.",neurological
"My lower back hurts when I bend or lift things.",musculoskeletal
"I cannot sleep and I feel anxious most nights.",mental_health_sleep
```

## Data Split

The dataset was split into training and test sets using stratified sampling.

| Split | Rows |
|---|---:|
| Training set | 67 |
| Test set | 23 |
| Total | 90 |

Split settings:

```text
Test size: 0.25
Random seed: 712
Stratified by label
```

## Embedding Model

The project uses the following embedding model:

```text
nicher92/saga-embed_v1
```

The embedding model converts each symptom description into a dense vector representation.

The embedding model was used as a frozen text encoder. Only the downstream classifiers were trained for this assignment.

## Classifiers

Four classifiers were trained on the same embedding vectors:

- Logistic Regression
- Linear SVM
- KNN with cosine distance
- Random Forest

Pipeline:

```text
symptom description -> frozen embedding model -> embedding vector -> classifier -> predicted category
```

The best classifier was Logistic Regression, which was saved as:

```text
model_phase2.joblib
```

## Evaluation

The classifiers were evaluated on the held-out test set.

Evaluation results are stored in:

```text
metrics.json
predictions_phase2.csv
```

## Classifier Comparison

| Classifier | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.957 | 0.952 | 0.957 |
| Linear SVM | 0.957 | 0.952 | 0.957 |
| KNN cosine | 0.783 | 0.763 | 0.764 |
| Random Forest | 0.913 | 0.903 | 0.909 |

Logistic Regression and Linear SVM achieved the same top score. Logistic Regression was selected for the demo because it provides probability scores through `predict_proba`, which makes the demo output more informative.

Macro F1 is especially important because it measures whether the classifier performs consistently across all categories, not only on the most common label.

## Example Prediction

Input:

```text
I have a fever, dry cough, and sore throat.
```

Output:

```text
Predicted information category: respiratory
```

Suggested retrieval focus:

```text
cough, fever, sore throat public health guidance
```

## Error Analysis

Some errors are expected because short symptom descriptions can contain signals from more than one category.

Example:

```text
I feel tired, dizzy, and cannot sleep.
```

This could be difficult because `dizzy` may suggest the `neurological` category, while `cannot sleep` may suggest the `mental_health_sleep` category.

These overlapping cases show why the model should be used only for broad information routing, not for diagnosis or medical decision-making.

## Hugging Face Demo

A working Hugging Face Space demo is available here:

```text
TODO: add your Hugging Face Space link
```

The demo allows users to enter a short symptom description and returns:

- predicted category
- suggested retrieval focus
- category probability scores
- a safety notice

## Links

| Resource | Link |
|---|---|
| Hugging Face Dataset | https://huggingface.co/datasets/charlie0831/english-symptom-routing |
| Hugging Face Model | https://huggingface.co/charlie0831/symptom-routing-embedding-classifiers |
| Hugging Face Space Demo | https://huggingface.co/spaces/charlie0831/symptom-routing-embedding-demo |
| GitHub Repository | TODO |

## Repository Files

| File | Description |
|---|---|
| `app.py` | Gradio web demo for predicting routing categories. |
| `preprocess.py` | Reproducible training and evaluation script. |
| `requirements.txt` | Python dependencies needed to run the project. |
| `symptom_routing_expanded_dataset.csv` | Custom symptom routing dataset with 90 examples. |
| `train.csv` | Training split generated from the custom dataset. |
| `test.csv` | Test split generated from the custom dataset. |
| `model_phase2.joblib` | Best trained classifier used by the demo. |
| `all_classifiers_phase2.joblib` | All trained classifiers from the comparison experiment. |
| `metrics.json` | Evaluation metrics for all classifiers. |
| `predictions_phase2.csv` | Test set predictions from the selected model. |
| `report.md` | Short academic report for the assignment. |

## How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Reproduce training and evaluation:

```bash
python preprocess.py
```

Run the Gradio app:

```bash
python app.py
```

Then open the local URL shown in the terminal.

## Intended Use

This project is intended for:

- information retrieval education
- text classification experiments
- embedding-based classifier demonstrations
- public health information routing prototypes

## Out-of-Scope Use

This project must not be used for:

- medical diagnosis
- treatment recommendation
- medication advice
- emergency guidance
- clinical triage
- real patient decision-making

If someone has severe, worsening, or urgent symptoms, they should contact a qualified medical professional or emergency service.

## Limitations

The dataset is small and created for an educational assignment. It does not represent the full variety of real-world symptom descriptions. The model may make mistakes, especially when a text contains symptoms from multiple categories.

The labels are broad information categories and should not be interpreted as diseases or clinical conditions.

The classifier depends on the quality of the embeddings and the training examples. More data, more diverse writing styles, and further evaluation would be needed before considering any real-world use.

## AI Tool Reflection

AI coding tools were used to support coding, documentation, dataset formatting, training script development, model card writing, and report drafting. The outputs were manually checked and edited, especially the safety statements, label definitions, evaluation results, and project limitations.

Using AI tools made it faster to build the project structure and write code, but it was still necessary to understand each part of the pipeline. In particular, I checked that the system was framed as information routing rather than diagnosis, and that the embedding model was used only as a frozen encoder while the downstream classifiers were trained for the assignment.

## License

This project is released under the MIT License.


