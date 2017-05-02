import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


class RandomForest(object):

    def __init__(self, model_path):
        self._model_path = model_path
        self._clf = None

    def train(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.2,
                                                            random_state=44)
        print('train ...')
        self._clf = RandomForestClassifier(n_estimators=20, max_features='auto')
        self._clf.fit(x_train, y_train)

        score = self._clf.score(x_test, y_test)
        print('Score: ', score)

        joblib.dump(self._clf, self._model_path)
        print('model saved.')

    def _load_clf(self):
        if self._clf is None:
            self._clf = joblib.load(self._model_path)

    def predict(self, feature):
        self._load_clf()
        return self._clf.predict(feature)

    def predict_proba(self, feature):
        self._load_clf()
        return self._clf.predict_proba(feature)

    def predict_proba_top_k(self, feature, k=5):
        result = self.predict_proba(feature)

        x = np.array(result[0])
        argsort_index_r = np.argsort(x)[::-1]
        sort_value_r = np.sort(x)[::-1]
        index = []
        value = []
        for i in range(k):
            if i == k or i > len(sort_value_r):
                break
            index.append(argsort_index_r[i])
            value.append(sort_value_r[i])
        return index, value
