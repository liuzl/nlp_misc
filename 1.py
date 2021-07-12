from transformers import pipeline
classifier = pipeline("sentiment-analysis")
ret = classifier('We are very happy to show you the 🤗 Transformers library.')
print(ret)
