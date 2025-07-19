import evaluate

def evaluate_accuracy(predictions, references):
    metric = evaluate.load("accuracy")
    res = metric.compute(predictions=predictions, references=references)
    return res