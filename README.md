# Symptom Description Routing for Public Health Information Retrieval

This project trains an embedding-based text classifier for routing short English symptom descriptions into broad public health information categories. The purpose is to test whether text embeddings can support query routing before information retrieval.

This project is part of an Information Retrieval assignment. It includes a custom dataset, a trained classifier, evaluation results, and a working Hugging Face demo.

**Important safety notice:**  
This model is for teaching information retrieval and text classification. It does not provide medical diagnosis, treatment advice, or emergency guidance.

## Research Question

Can text embeddings classify short symptom descriptions into broad health-information categories for routing users toward relevant public health resources?

## Task Definition

Given a short symptom description, the system predicts one broad health-information category. The predicted category can be used as a routing signal for public health information retrieval.

The system is designed to:

- classify short English symptom descriptions
- route user queries toward broad public health information categories
- demonstrate embedding-based text classification
- compare classifier performance using a custom dataset

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

