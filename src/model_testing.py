from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import pandas as pd
from prep import DataPrep


def model_evaluation_cv(estimator, X, y, cv=5, scoring='accuracy', return_train_score=False):
    """
    Evaluate results of Cross Validation
    """
    cv_results = cross_validate(estimator=estimator, X=X, y=y, cv=cv, scoring=scoring, return_train_score=return_train_score)
    
    test_scores = cv_results['test_score']
    avg_test_score = test_scores.mean()
    train_scores = cv_results['train_score']
    avg_train_score = train_scores.mean()

    for i, j in cv_results.items():
        print(i, j)
    print('-----')
    print(f'Average cross-validation test score: {avg_test_score}')
    print(f'Average cross-validation train score: {avg_train_score}')


def make_predictions(fitted_pipeline, submission_name):
    # Read kaggle test data from disk
    X_test = pd.read_csv('data/test.csv')
    
    # Prep test data
    X_test_prepped = X_test.drop(columns='id')
    X_test_prepped = DataPrep().run(X_test_prepped)

    # Use fitted model pipeline to predict values on test data
    submission = pd.DataFrame({
        'id': list(X_test['id']),
        'return': fitted_pipeline.predict(X_test_prepped)
    })

    # Write results to csv file
    submission.to_csv(f'./submissions/{submission_name}', index=False)
    

def evaluate_confusion_matrix(estimator: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
    estimator.fit(X_train,  y_train)
    y_pred_train = estimator.predict(X_train)
    y_pred_val = estimator.predict(X_test)

    print(f'Training Accuracy: {round(accuracy_score(y_train, y_pred_train), 4)}')
    print(f'Test Accuracy: {round(accuracy_score(y_test, y_pred_val), 4)}')
    print(f'Precision: {round(precision_score(y_test, y_pred_val), 4)}')
    print(f'Recall: {round(recall_score(y_test, y_pred_val), 4)}')
    print(f'F1 Score: {round(f1_score(y_test, y_pred_val), 4)}')

    print('')

    print('---CONFUSION MATRIX---')
    cm = confusion_matrix(y_test, y_pred_val, labels=estimator.classes_)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=estimator.classes_).plot()
    print(f'True Positives: {cm[1][1]}')
    print(f'True Negatives: {cm[0][0]}')
    print(f'False Positives: {cm[0][1]}')
    print(f'False Negatives: {cm[1][0]}')
    cmd
