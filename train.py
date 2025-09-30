from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib, os

# Sample training data
texts = ["I love this", "This is bad", "Awesome work", "I hate it"]
labels = ["positive", "negative", "positive", "negative"]

# Train
model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(texts, labels)

# Save
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.joblib")
print("✅ Model saved at model/model.joblib")
