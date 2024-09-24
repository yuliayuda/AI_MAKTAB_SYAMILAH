from datasets import load_metric

def evaluate_model(model, dataset):
    metric = load_metric("accuracy")

    predictions = model.predict(dataset['validation'])
    preds = predictions.predictions.argmax(-1)
    metric.add_batch(predictions=preds, references=dataset['validation']['labels'])

    result = metric.compute()
    return result
