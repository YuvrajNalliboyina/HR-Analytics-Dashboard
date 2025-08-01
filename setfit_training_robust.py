from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
import os

# ✅ Load & preprocess dataset
df = pd.read_csv("nexttrain.csv")
df["label"] = df["label"].str.strip().str.lower()
df = df[["complaint_what_happened", "label"]].dropna()

print(f"Dataset shape: {df.shape}")
print(f"Unique labels: {df['label'].unique()}")

# ✅ Encode labels
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])
joblib.dump(le, "setfit_label_encoder.pkl")

# ✅ Create Dataset - Alternative approach to avoid Column object issues
# Method 1: Convert DataFrame to dict first, then to Dataset
data_dict = {
    "text": df["complaint_what_happened"].tolist(),
    "label": df["label_encoded"].tolist()
}

dataset = Dataset.from_dict(data_dict)
dataset = dataset.train_test_split(test_size=0.15, seed=42)

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

# ✅ Load SetFit model
model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-mpnet-base-v2",
    use_differentiable_head=True,  # This can help with some compatibility issues
    head_params={"out_features": len(le.classes_)}  # Specify number of classes
)

# ✅ Define TrainingArguments with corrected parameter name
args = TrainingArguments(
    output_dir="./setfit_results",
    num_epochs=5,
    batch_size=16,
    eval_strategy="epoch",  # Fixed: was evaluation_strategy
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    seed=42
)

# ✅ Create Trainer with error handling
try:
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        metric="accuracy"
    )
    
    # ✅ Train
    print("Starting training...")
    trainer.train()
    
    # ✅ Evaluate
    print("Evaluating model...")
    metrics = trainer.evaluate()
    print("📊 Final Metrics:", metrics)
    
    # ✅ Save model
    print("Saving model...")
    model.save_pretrained("./setfit_sentiment_model")
    print("✅ Model saved successfully!")
    
    # ✅ Test inference
    print("\n🧪 Testing inference...")
    test_texts = dataset["test"]["text"][:3]  # Test on first 3 examples
    predictions = model(test_texts)
    
    for i, (text, pred) in enumerate(zip(test_texts, predictions)):
        predicted_label = le.inverse_transform([pred])[0]
        print(f"Text {i+1}: '{text[:100]}...'")
        print(f"Predicted: {predicted_label}")
        print()

except Exception as e:
    print(f"❌ Error during training: {e}")
    print("Trying alternative approach...")
    
    # Alternative approach: Disable model card generation
    import setfit.model_card
    original_method = setfit.model_card.ModelCard.set_widget_examples
    
    def patched_set_widget_examples(self, dataset):
        """Patched version that handles Column objects properly"""
        try:
            if hasattr(dataset, 'to_pandas'):
                # If it's a Dataset object, convert to pandas first
                df_temp = dataset.to_pandas()
                if 'text' in df_temp.columns:
                    samples = df_temp['text'].tolist()[:10]  # Convert to list
                else:
                    samples = []
            else:
                samples = []
            
            self.widget = [{"text": sample} for sample in samples]
            
            if samples:
                samples.sort(key=len)
                self.predict_example = samples[0] if samples else ""
        except Exception:
            # If all else fails, set empty widget
            self.widget = []
            self.predict_example = ""
    
    # Apply the patch
    setfit.model_card.ModelCard.set_widget_examples = patched_set_widget_examples
    
    # Retry training with patched model card
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        metric="accuracy"
    )
    
    trainer.train()
    metrics = trainer.evaluate()
    print("📊 Final Metrics:", metrics)
    model.save_pretrained("./setfit_sentiment_model")
    print("✅ Model saved successfully with patch!")