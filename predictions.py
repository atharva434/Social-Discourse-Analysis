import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_text(text, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    pred = logits.argmax().item()
    return "Polarized" if pred == 1 else "Neutral"