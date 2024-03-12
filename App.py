# from flask import Flask, render_template, request
# import pickle # Replace with the actual module where your model code resides
# import numpy as np

# app = Flask(__name__)

# model=pickle.load(open('model.pkl','rb'))
# @app.route('/')
# def home():
#     #return render_template('form.html')  # Assuming your HTML file is named 'index.html'
#      return render_template('index.html')

# @app.route('/predict', methods=['POST','GET'])
# def predict():
#     if request.method == 'POST':
 
#         int_features=[int(x) for x in request.form.values()]
#         final=[np.array(int_features)]

#         # Call your machine learning model function
#         prediction = model.predict(final)  
#         # if isinstance(prediction, list) and prediction[0] == "Low":
#         #     prediction[0] = "5 To 10 years"
#         # prediction="5 To 10 years"
#         return render_template('output.html', pred=prediction)

# if __name__ == '__main__':
#     app.run()

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the machine learning model
model = pickle.load(open('model.pkl', 'rb'))

# Define a dictionary to map prediction values to survival periods
prediction_mapping = {
    "Low": "There is 55% chances that patient can survive for 5 years",
    "Medium": "There is 25% chances that patient can survive for 5 years",
    "High": "There is 5% chances that patient can survive for 5 years"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the features from the form
        int_features = [int(x) for x in request.form.values()]
        final = [np.array(int_features)]

        # Call your machine learning model function to make a prediction
        prediction = model.predict(final)[0]  # Assuming only one prediction is made

        # Map the prediction value to the corresponding survival period
        survival_period = prediction_mapping.get(prediction, "Unknown")

        return render_template('output.html', pred=survival_period)

if __name__ == "__main__":
    app.run(debug=True)
