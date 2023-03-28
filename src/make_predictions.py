from sklearn.preprocessing import OrdinalEncoder, Normalizer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import numpy as np

from create_datasets import get_training
from prep import DataPrep
from model_testing import make_predictions

# Get X and y data
X, y = get_training()

# Remove price outliers
training_data = X.join(y)
training_data = training_data[training_data['price'] <= 600].reset_index(drop=True)
X = training_data.drop(columns='return')
y = training_data['return']

# Prepare and clean X Training data
X_prepped = DataPrep().run(X)

# Create lists of numerical and categorical columns in X data
numeric_cols = X_prepped.select_dtypes(include=np.number).columns
categorical_cols = X_prepped.select_dtypes(exclude=np.number).columns

# Initialize the preprocessor
preprocessor = ColumnTransformer([
    ('categorical_features', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]),categorical_cols),
    ('numeric_features', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('normalize', Normalizer(norm='max'))
        ]), numeric_cols)
    ])

# Tuned Classifier
clf = XGBClassifier(
    n_estimators=300,
    objective='binary:logistic',
    tree_method='hist',
    learning_rate=0.1,
    max_depth=4,
    booster='gbtree',
    min_child_weight=5,
    gamma=0,
    subsample=1,
    scale_pos_weight=1,
    eval_metric='error'
)

# ML Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', clf)])

# Fit base pipeline to training data
pipeline.fit(X_prepped, y)

make_predictions(pipeline, 'submission10_xgboost.csv')