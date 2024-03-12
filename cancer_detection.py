import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import pickle

# Load the dataset
df = pd.read_csv("lungcancerdataset.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Separating data and labels
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Initialize and train the model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
gradient_boosting_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gnb = GaussianNB()
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', random_forest_model),
        ('gb', gradient_boosting_model),
        ('gnb',gnb)
    ],
    voting='soft'  # Soft voting takes into account the probability estimates
)
ensemble_model.fit(X_train, Y_train)

# Evaluate the model on training data
gnb_train_pred = ensemble_model.predict(X_train)
gnb_train_accuracy = accuracy_score(Y_train, gnb_train_pred)

# Evaluate the model on testing data
gnb_test_pred = ensemble_model.predict(X_test)
gnb_test_accuracy = accuracy_score(Y_test, gnb_test_pred)

# Save the trained model using pickle
pickle.dump(ensemble_model, open('model.pkl', 'wb'))

# Load the model from the pickle file
model = pickle.load(open('model.pkl', 'rb'))
