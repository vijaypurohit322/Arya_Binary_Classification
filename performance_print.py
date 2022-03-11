import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,log_loss
from prettytable import PrettyTable

print('Printing Model Performance...')

# loading the train data
df = pd.read_csv('data/training_set.csv',index_col=0)

# Splitting the data
X = df.drop(['Y'],axis=1)
y = df['Y']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Feature Selection using RandomForest
clf = RandomForestClassifier(100, max_depth=None, n_jobs=-1)
clf.fit(X_train,y_train)
feature_importance = clf.feature_importances_
# Ranking the features with their respect to feature importances
fi = sorted(zip(X.columns,feature_importance),key=lambda x: x[1], reverse=True)
# Extracting Top 30 features
top_features = [x[0] for x in fi[:30]]

# Selecting the top features from data
X_train_dash = X_train[top_features]
X_test_dash = X_test[top_features]

# Normalization
# Using StandardScaler to normalize our data
scaler = StandardScaler()
scaler.fit(X_train_dash)
# transform
X_train_dash = pd.DataFrame(scaler.transform(X_train_dash),columns=X_train_dash.columns)
X_test_dash = pd.DataFrame(scaler.transform(X_test_dash),columns=X_test_dash.columns)

# Using PrettyTable for printing performance of each model
table = PrettyTable()
table.field_names = ["Model","Train Logloss", "Validation Logloss", "Train AUC", "Validation AUC"]

# Using Random Model
y_train_prob = np.random.rand(len(X_train_dash))
y_test_prob = np.random.rand(len(X_test_dash))

table.add_row(["Random", log_loss(y_train,y_train_prob),log_loss(y_test,y_test_prob), roc_auc_score(y_train, y_train_prob),roc_auc_score(y_test, y_test_prob)])


# Using KNN
# Using 30-NN for predictions
classifier = KNeighborsClassifier(n_neighbors=30) 
classifier.fit(X_train_dash, y_train)

y_train_pred = classifier.predict(X_train_dash)
y_train_prob = classifier.predict_proba(X_train_dash)[:,1]
y_test_pred = classifier.predict(X_test_dash)
y_test_prob = classifier.predict_proba(X_test_dash)[:,1]

table.add_row(["KNN",log_loss(y_train,y_train_prob), log_loss(y_test,y_test_prob),roc_auc_score(y_train, y_train_prob), roc_auc_score(y_test, y_test_prob)])

# Using Gaussian Naive Bayes
classifier = GaussianNB()
classifier.fit(X_train_dash, y_train)

y_train_pred = classifier.predict(X_train_dash)
y_train_prob = classifier.predict_proba(X_train_dash)[:,1]
y_test_pred = classifier.predict(X_test_dash)
y_test_prob = classifier.predict_proba(X_test_dash)[:,1]

table.add_row(["Naive Bayes", log_loss(y_train,y_train_prob), log_loss(y_test,y_test_prob),roc_auc_score(y_train, y_train_prob), roc_auc_score(y_test, y_test_prob)])

# Using Logistic Regression with l2 norm
classifier = LogisticRegression(C=1, penalty='l2', max_iter=250, random_state=42)
classifier.fit(X_train_dash, y_train)

y_train_pred = classifier.predict(X_train_dash)
y_train_prob = classifier.predict_proba(X_train_dash)[:,1]
y_test_pred = classifier.predict(X_test_dash)
y_test_prob = classifier.predict_proba(X_test_dash)[:,1]

table.add_row(["Logistic Regression", log_loss(y_train,y_train_prob), log_loss(y_test,y_test_prob),roc_auc_score(y_train, y_train_prob), roc_auc_score(y_test, y_test_prob)])


# Using DecisionTree
classifier = DecisionTreeClassifier(criterion='gini',min_samples_split=3,random_state=42)
classifier.fit(X_train_dash, y_train)

y_train_pred = classifier.predict(X_train_dash)
y_train_prob = classifier.predict_proba(X_train_dash)[:,1]
y_test_pred = classifier.predict(X_test_dash)
y_test_prob = classifier.predict_proba(X_test_dash)[:,1]

table.add_row(["DecisionTree",log_loss(y_train,y_train_prob), log_loss(y_test,y_test_prob),roc_auc_score(y_train, y_train_prob), roc_auc_score(y_test, y_test_prob)])


# Using RandomForest
classifier = RandomForestClassifier(n_estimators=500,
                                      max_depth=None,
                                      min_samples_split=2,
                                      n_jobs=-1,
                                      class_weight='balanced',
                                      random_state=42)
classifier.fit(X_train_dash, y_train)

y_train_pred = classifier.predict(X_train_dash)
y_train_prob = classifier.predict_proba(X_train_dash)[:,1]
y_test_pred = classifier.predict(X_test_dash)
y_test_prob = classifier.predict_proba(X_test_dash)[:,1]

table.add_row(["RandomForest", log_loss(y_train,y_train_prob), log_loss(y_test,y_test_prob),roc_auc_score(y_train, y_train_prob), roc_auc_score(y_test, y_test_prob)])


# Using Xgboost
classifier = XGBClassifier(n_estimators=500,
                           max_depth=5,
                           learning_rate=0.15,
                           colsample_bytree=1,
                           subsample=1,
                           reg_alpha = 0.3,
                           gamma=10,
                           n_jobs=-1,
                           eval_metric='logloss',
                           use_label_encoder=False)

classifier.fit(X_train_dash, y_train)

y_train_pred = classifier.predict(X_train_dash)
y_train_prob = classifier.predict_proba(X_train_dash)[:,1]
y_test_pred = classifier.predict(X_test_dash)
y_test_prob = classifier.predict_proba(X_test_dash)[:,1]

table.add_row(["Xgboost", log_loss(y_train,y_train_prob), log_loss(y_test,y_test_prob),roc_auc_score(y_train, y_train_prob), roc_auc_score(y_test, y_test_prob)])
print(table)