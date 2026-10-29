from transformers import pipeline

def analyze_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = sentiment_pipeline(text)[0]
    return result

positive_sentence = "I absolutely love this new phone, it's incredible!"
positive_result = analyze_sentiment(positive_sentence)
print(f"Sentence: '{positive_sentence}'")
print(f"Sentiment: {positive_result['label']} (Score: {positive_result['score']:.4f})")
print("-" * 20)

negative_sentence = "This movie was a complete waste of time, I'm so disappointed."
negative_result = analyze_sentiment(negative_sentence)
print(f"Sentence: '{negative_sentence}'")
print(f"Sentiment: {negative_result['label']} (Score: {negative_result['score']:.4f})")
print("-" * 20)

neutral_sentence = "The food was okay, but the service was terrible."
neutral_result = analyze_sentiment(neutral_sentence)
print(f"Sentence: '{neutral_sentence}'")
print(f"Sentiment: {neutral_result['label']} (Score: {neutral_result['score']:.4f})")
