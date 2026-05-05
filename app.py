import joblib
import gradio as gr
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "nicher92/saga-embed_v1"
CLASSIFIER_PATH = "model_phase2.joblib"


RETRIEVAL_FOCUS = {
    "respiratory": "Look for public health information about cough, sore throat, fever, wheezing, and breathing-related symptoms.",
    "gastrointestinal": "Look for public health information about stomach pain, nausea, vomiting, diarrhea, bloating, and digestion-related symptoms.",
    "skin": "Look for public health information about rashes, itching, swelling, hives, dry skin, and skin irritation.",
    "mental_health_sleep": "Look for public health information about sleep problems, stress, anxiety, low mood, restlessness, and wellbeing support.",
    "neurological": "Look for public health information about headache, dizziness, numbness, tingling, balance problems, and vision-related symptoms.",
    "musculoskeletal": "Look for public health information about back pain, joint pain, muscle soreness, stiffness, injury, and movement-related pain.",
}


DISCLAIMER = (
    "This demo is for an Information Retrieval course project. "
    "It does not provide medical diagnosis, treatment advice, emergency triage, "
    "or personalized medical guidance."
)


embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
classifier = joblib.load(CLASSIFIER_PATH)


def predict_category(symptom_description):
    text = symptom_description.strip()
    if not text:
        return "Please enter a short symptom description.", "", ""

    embedding = embedding_model.encode([text], normalize_embeddings=True)
    predicted_label = classifier.predict(embedding)[0]

    probability_text = "Probability scores are not available for this classifier."
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(embedding)[0]
        labels = classifier.classes_
        ranked = sorted(
            zip(labels, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )
        probability_text = "\n".join(
            f"{label}: {score:.3f}" for label, score in ranked
        )

    result = f"Predicted information category: {predicted_label}"
    retrieval_focus = RETRIEVAL_FOCUS.get(
        predicted_label,
        "Look for relevant public health information from reliable sources.",
    )

    return result, retrieval_focus, probability_text


with gr.Blocks(title="English Symptom Routing Demo") as demo:
    gr.Markdown("# English Symptom Routing Demo")
    gr.Markdown(
        "Enter a short English symptom description. The demo routes it to a broad "
        "public health information category for retrieval."
    )

    symptom_input = gr.Textbox(
        label="Symptom description",
        placeholder="Example: I have a dry cough and a sore throat.",
        lines=4,
    )

    predict_button = gr.Button("Classify")

    category_output = gr.Textbox(label="Prediction")
    retrieval_output = gr.Textbox(label="Suggested retrieval focus")
    probability_output = gr.Textbox(label="Category scores")

    gr.Markdown(f"**Important:** {DISCLAIMER}")

    examples = gr.Examples(
        examples=[
            ["I have a dry cough and a sore throat."],
            ["My stomach hurts after eating and I feel nauseous."],
            ["I have an itchy red rash on my arm."],
            ["I cannot sleep and feel anxious at night."],
            ["I have a strong headache and feel dizzy."],
            ["My lower back hurts when I bend."],
        ],
        inputs=symptom_input,
    )

    predict_button.click(
        fn=predict_category,
        inputs=symptom_input,
        outputs=[category_output, retrieval_output, probability_output],
    )

    symptom_input.submit(
        fn=predict_category,
        inputs=symptom_input,
        outputs=[category_output, retrieval_output, probability_output],
    )


if __name__ == "__main__":
    demo.launch()
