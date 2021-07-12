from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
model_name = "liam168/c4-zh-distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
ts_texts = ["女人做得越纯粹，皮肤和身材就越好", "我喜欢篮球"]
r1 = classifier(ts_texts[0])
r2 = classifier(ts_texts[1])
print(r1, r2)
