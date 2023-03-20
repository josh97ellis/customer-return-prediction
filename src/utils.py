from sklearn.model_selection import cross_validate
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
