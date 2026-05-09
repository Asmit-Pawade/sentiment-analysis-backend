import pickle
import numpy as np
from deep_translator import GoogleTranslator

print("Loading ML model and vectorizer...")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Label mapping
label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

#  Keyword dictionaries

sad_words = [
    "sad", "unhappy", "depressed", "cry", "crying", "tears",
    "lonely", "alone", "heartbroken", "broken", "miserable",
    "hopeless", "helpless", "upset", "hurt", "pain", "sorrow",
    "grief", "regret", "lost", "empty", "worthless", "disappointed"
]

anger_words = [
    "angry", "mad", "furious", "rage", "hate", "annoyed",
    "irritated", "frustrated", "pissed", "outraged",
    "fed up", "aggressive", "yelling", "shouting",
    "temper", "hostile", "fuming"
]

fear_words = [
    "scared", "afraid", "fear", "terrified", "panic",
    "nervous", "anxious", "worried", "tense",
    "frightened", "stress", "stressed", "unsafe"
]

disgust_words = [
    "disgust", "disgusting", "gross", "nasty", "dirty",
    "filthy", "sick", "awful", "horrible", "terrible",
    "yuck", "repulsive", "vomit", "toxic"
]

joy_words = [
    "happy", "joy", "excited", "amazing", "awesome",
    "great", "fantastic", "love", "wonderful",
    "delighted", "pleased", "smile", "glad",
    "enjoy", "fun", "blessed", "grateful"
]

surprise_words = [
    "wow", "unexpected", "surprised", "shocked",
    "unbelievable", "amazed", "astonished"
]


def predict_sentiment(text: str):
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")

    #  Step 1: Translation
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
    except:
        translated = text

    text_lower = translated.lower()

    #  Step 2: Vectorization
    text_vector = vectorizer.transform([translated])

    #  Step 3: ML Prediction
    raw = model.predict(text_vector)[0]
    raw = int(raw) if isinstance(raw, (np.integer, int)) else raw
    prediction = label_map.get(raw, "neutral")

    #  Step 4: RULE-BASED SENTIMENT OVERRIDE (CRITICAL FIX)

    if any(word in text_lower for word in anger_words + sad_words + fear_words + disgust_words):
        prediction = "negative"

    elif any(word in text_lower for word in joy_words):
        prediction = "positive"

    #  Step 5: Confidence
    try:
        probs = model.predict_proba(text_vector)[0]
        confidence = float(np.max(probs))
        if not np.isfinite(confidence):
            confidence = 1.0
    except:
        confidence = 1.0

    #  Step 6: Emotion Detection

    if prediction == "positive":

        if any(word in text_lower for word in joy_words):
            emotions = {"joy": 80, "surprise": 10, "neutral": 5, "sadness": 2, "anger": 1, "fear": 1, "disgust": 1}

        elif any(word in text_lower for word in surprise_words):
            emotions = {"joy": 20, "surprise": 70, "neutral": 5, "sadness": 2, "anger": 1, "fear": 1, "disgust": 1}

        else:
            emotions = {"joy": 60, "surprise": 20, "neutral": 10, "sadness": 3, "anger": 2, "fear": 2, "disgust": 3}

    elif prediction == "negative":

        if any(word in text_lower for word in sad_words):
            emotions = {"sadness": 75, "anger": 8, "fear": 6, "disgust": 4, "joy": 1, "surprise": 1, "neutral": 5}

        elif any(word in text_lower for word in anger_words):
            emotions = {"anger": 75, "sadness": 8, "fear": 5, "disgust": 5, "joy": 1, "surprise": 1, "neutral": 5}

        elif any(word in text_lower for word in fear_words):
            emotions = {"fear": 75, "sadness": 8, "anger": 5, "disgust": 3, "joy": 1, "surprise": 5, "neutral": 5}

        elif any(word in text_lower for word in disgust_words):
            emotions = {"disgust": 72, "anger": 10, "fear": 5, "sadness": 5, "joy": 1, "surprise": 2, "neutral": 5}

        else:
            emotions = {"sadness": 40, "anger": 30, "fear": 10, "disgust": 5, "joy": 2, "surprise": 3, "neutral": 10}

    else:
        emotions = {"neutral": 70, "joy": 5, "surprise": 5, "sadness": 5, "anger": 5, "fear": 5, "disgust": 5}

    #  Step 7: Normalize
    total = sum(emotions.values()) or 1
    emotions = {k: round((v / total) * 100, 1) for k, v in emotions.items()}

    # 🔹 Step 8: Final Output
    return {
        "sentiment": prediction.upper(),
        "confidence": round(confidence * 100, 1),
        "top_emotion": max(emotions, key=emotions.get),
        "all_emotions": emotions,
        "original_text": text,
        "translated_text": translated,
        "word_count": len(text.split()),
        "char_count": len(text),
    }