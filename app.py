import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, f1_score,RocCurveDisplay
import matplotlib.pyplot as plt


# Load data

def load_data(dataset):
    if dataset == "heart":
        data1 = pd.read_csv("heart.csv")
        columns_to_drop = ['ca', 'thal','sex','fbs','restecg','slope','trestbps']
        data1.drop(columns=columns_to_drop, inplace=True)
    elif dataset == "heart_statlog":
        data1 = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")
        data1 = data1.rename(columns={
            'age': 'age',
            'sex': 'sex',
            'chest pain type': 'cp',
            'resting bp s': 'trestbps',
            'cholesterol': 'chol',
            'fasting blood sugar': 'fbs',
            'resting ecg': 'restecg',
            'max heart rate': 'thalach',
            'exercise angina': 'exang',
            'oldpeak': 'oldpeak',
            'ST slope': 'slope',
            'target': 'target'
        })
        columns_to_drop = ['sex', 'fbs', 'restecg', 'slope', 'trestbps']
        data1.drop(columns=columns_to_drop, inplace=True)

    return data1

# Sidebar for dataset selection
selected_dataset = st.sidebar.selectbox("Select Dataset", ["heart", "heart_statlog"])

data = st.cache_data(load_data)(selected_dataset)

# Extract features and target variable

X = data.drop(columns='target', axis=1)
Y = data['target']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
st.title('Heart Disease Prediction')


st.sidebar.header('User Input Features')


age = st.sidebar.slider('Age', min_value=0, max_value=100, value=50)

cp = st.sidebar.slider('Chest Pain Type', min_value=0, max_value=3, value=1)

chol = st.sidebar.slider('Cholesterol', min_value=100, max_value=600, value=200)

thalach = st.sidebar.slider('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
exang = st.sidebar.radio('Exercise Induced Angina', ['No', 'Yes'])
oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=6.2, value=2.0)



exang = 1 if exang == 'Yes' else 0




# Create a new data point using user input
user_input = pd.DataFrame({
    'age': [age],

    'cp': [cp],

    'chol': [chol],


    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],

})

# Model selection
model_type = st.sidebar.radio('Select Model', ('Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM','KNN',"Naive Baye's","GBM"))

# Model training and prediction
if model_type == 'Logistic Regression':
    model = LogisticRegression()
elif model_type == 'Decision Tree':
    model = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=6)
elif model_type == 'Random Forest':
    model = RandomForestClassifier(n_estimators=15, random_state=3)
elif model_type == 'SVM':
    model = SVC(kernel="linear",C=0.1)
elif model_type == 'KNN':
    model = KNeighborsClassifier(n_neighbors=7)
elif model_type == "Naive Baye's":
    model =  GaussianNB()
elif model_type == "GBM":
    if selected_dataset == "heart":
      model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_split=5, min_samples_leaf=2, subsample=0.8, random_state=42)
    else:
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, min_samples_split=10, min_samples_leaf=5, subsample=0.8, random_state=42)

model.fit(X_train_scaled, Y_train)
Y_pred_test = model.predict(X_test_scaled)
Y_pred_user_input = model.predict(user_input)

# Calculate evaluation metrics for the actual test set
accuracy_test = accuracy_score(Y_test, Y_pred_test)
precision_test = precision_score(Y_test, Y_pred_test)
f1_test = f1_score(Y_test, Y_pred_test)

# Calculate evaluation metrics for the user input
# Note: This assumes that you have true labels for the user input, which might not be the case in practice.
# For simplicity, I'm omitting this part, but you can compare the predicted label with the actual label if available.
accuracy_user_input = None
precision_user_input = None
f1_user_input = None

# Display prediction for user input
if Y_pred_user_input == 1:
    prediction_label = "Heart Disease Present"
    prediction_emoji = "❤️"
else:
    prediction_label = "No Heart Disease Detected"
    prediction_emoji = "✅"
st.write('Prediction:', f"{prediction_label} {prediction_emoji}")

# Display evaluation metrics for actual test set
st.info(f"{model_type} Metrics for Actual Test Set:")
st.write("Accuracy:", accuracy_test)
st.write("Precision:", precision_test)
st.write("F1 Score:", f1_test)

st.subheader('ROC Curve for Actual Test Set')
fig, ax = plt.subplots()
RocCurveDisplay.from_estimator(model, X_test, Y_test, ax=ax)
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
st.pyplot(fig)
