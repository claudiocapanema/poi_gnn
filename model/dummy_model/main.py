import numpy as np
from sklearn.dummy import DummyClassifier
import sklearn.metrics as skm
import pandas as pd


if __name__ == "__main__":

    methods = []
    scores = []
    reports = []

    strategy = ['most_frequent', 'stratified', 'prior', 'uniform']
    for i in range(len(strategy)):

        a = 1434
        b = 4195
        c = 8511
        d = 3578
        e = 4311
        f = 3181
        g = 390
        y = [0] * a + [1] * b + [2] * c + [3] * d + [4] * d + [5] * e + [6] * f

        x = np.array([np.random.randint(1, 10) for j in range(len(y))])

        dummy_clf = DummyClassifier(strategy=strategy[i])
        dummy_clf.fit(x, y)
        predicted = dummy_clf.predict(x)
        score = dummy_clf.score(x, y)

        report = skm.classification_report(y_true=y, y_pred=predicted, output_dict=True)

        methods.append(strategy[i])
        scores.append(score)
        reports.append(str(report))

    df = pd.DataFrame({'method': methods, 'score': scores, 'metrics': reports})
    df.to_csv("metrics.csv", index=False)