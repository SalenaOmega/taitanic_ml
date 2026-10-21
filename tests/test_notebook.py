"""Regression tests execute the notebook's real definitions without running EDA."""
import ast
import contextlib
import io
import json
from pathlib import Path
import unittest
import warnings

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, cross_validate

ROOT = Path(__file__).resolve().parents[1]


def load_definitions():
    notebook = json.loads((ROOT / 'titanic_prediction.ipynb').read_text(encoding='utf-8'))
    namespace = {}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        definitions = [n for n in tree.body if isinstance(
            n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<notebook>', 'exec'), namespace)
    return notebook, namespace


class NotebookRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook, cls.ns = load_definitions()
        cls.train = pd.read_csv(ROOT / 'data/titanic/train.csv')
        cls.test = pd.read_csv(ROOT / 'data/titanic/test.csv')

    def fit_transformer(self):
        # A missing fold-local transformer is the original leakage failure.
        self.assertTrue('TitanicFeatures' in self.ns,
                      'Feature statistics must be fitted inside each Pipeline/CV fold')
        return self.ns['TitanicFeatures']().fit(self.train)

    def test_prediction_is_independent_of_batch_size_and_other_fits(self):
        transformer = self.fit_transformer()
        full = transformer.transform(self.test)
        single = transformer.transform(self.test.iloc[[0]])
        assert_frame_equal(single, full.iloc[[0]])
        other = self.train.copy()
        other['Fare'] *= 100
        self.ns['TitanicFeatures']().fit(other)
        assert_frame_equal(full, transformer.transform(self.test))

    def test_each_cv_fold_fits_only_its_training_rows(self):
        self.fit_transformer()
        raw = self.train.drop(columns='Survived')
        cv = KFold(n_splits=3, shuffle=True, random_state=7)
        result = cross_validate(self.ns['make_pipe'](DummyClassifier()), raw,
                                self.train.Survived, cv=cv, return_estimator=True)
        for (indices, _), estimator in zip(cv.split(raw), result['estimator']):
            transformer = estimator.named_steps['features']
            expected = raw.iloc[indices].Age.median()
            self.assertEqual(transformer.global_age_median_, expected)
            np.testing.assert_allclose(transformer.fare_edges_[1:-1],
                                       raw.iloc[indices].Fare.quantile([.25, .5, .75]))

    def test_extreme_fares_and_unseen_age_group_use_fitted_state(self):
        # Fit only Mr titles so Rare is an unseen group with a known fallback (28).
        transformer = self.ns['TitanicFeatures']().fit(
            self.train.assign(Name='Example, Mr. Known'))
        rows = self.test.iloc[:2].copy()
        rows['Fare'] = [0, 1_000_000]
        rows['Age'] = np.nan
        rows['Name'] = ['Unknown, Dona. Example', 'Unknown, Dona. Other']
        out = transformer.transform(rows)
        self.assertEqual(out.FareBin.tolist(), [0, 3])
        self.assertEqual(out.Age.tolist(), [28.0, 28.0])
        self.assertFalse(out.isna().any().any())
        for i in range(2):
            assert_frame_equal(transformer.transform(rows.iloc[[i]]), out.iloc[[i]])

    def test_classification_report_support_counts_actual_labels(self):
        cell = next(c for c in self.notebook['cells'] if c['cell_type'] == 'code'
                    and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                            and n.func.id == 'classification_report'
                            for n in ast.walk(ast.parse(''.join(c['source'])))))
        data = pd.DataFrame({'x': [0, 1, 2]})
        namespace = dict(self.ns, X_val=data, y_val=pd.Series([0, 0, 1]),
                         best_pipe=DummyClassifier(strategy='constant', constant=0).fit(data, [0, 0, 1]))
        output = io.StringIO()
        with warnings.catch_warnings(), contextlib.redirect_stdout(output):
            warnings.simplefilter('ignore')
            exec(''.join(cell['source']), namespace)
        line = next(line for line in output.getvalue().splitlines() if '遇难' in line)
        self.assertEqual(int(line.split()[-1]), 2,
                         'support must count actual labels, not predictions')


if __name__ == '__main__':
    unittest.main()
