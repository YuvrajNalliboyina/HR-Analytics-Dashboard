from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib

# ✅ Load & preprocess dataset
df = pd.read_csv("nexttrain.csv")
df["label"] = df["label"].str.strip().str.lower()
df = df[["complaint_what_happened", "label"]].dropna()

# ✅ Encode labels
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])
joblib.dump(le, "setfit_label_encoder.pkl")

# ✅ Create Dataset and rename columns for SetFit
# Convert to lists to avoid Column object issues
text_data = df["complaint_what_happened"].tolist()
label_data = df["label_encoded"].tolist()

# Create dataset from lists instead of pandas DataFrame
dataset_dict = {
    "text": text_data,
    "label": label_data
}
dataset = Dataset.from_dict(dataset_dict)
dataset = dataset.train_test_split(test_size=0.15, seed=42)

# ✅ Load SetFit model
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

# ✅ Define Trainer and TrainingArguments (fixed deprecated parameter)
args = TrainingArguments(
    num_epochs=5,
    batch_size=16,
    eval_strategy="epoch"  # Fixed: was evaluation_strategy
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    metric="accuracy"
)

# ✅ Train
trainer.train()

# ✅ Evaluate
metrics = trainer.evaluate()
print("📊 Accuracy:", metrics)

# ✅ Save model
model.save_pretrained("./setfit_sentiment_model")