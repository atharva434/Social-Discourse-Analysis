from transformers import AutoModelForSequenceClassification

def build_model(model_name, num_labels=2):
    """
    Instantiates the model architecture.
    """
    print(f"[INFO] Building model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    return model