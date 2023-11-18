from flask import Flask, render_template, request
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
model = pickle.load(open('model_decision_tree.pkl', 'rb'))


# Define the route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route to handle the form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Get the values entered in the form
    age = float(request.form['Age'])
    annual = float(request.form['AnnualSalary'])
    
    # Perform the addition
    result = model.predict([[age,annual]])
    print(result)


    # Render the result in a new template.
    return render_template('result.html', result=result)

if __name__ == '__main__':
     app.run(debug=True)