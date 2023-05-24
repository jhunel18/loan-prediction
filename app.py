from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Load the saved model
        model = joblib.load("loan_model.joblib")

        # Get the data from the form
        gender = int(request.form['gender'])
        married = int(request.form['married'])
        dependents = int(request.form['dependents'])
        self_employed = int(request.form['self_employed'])
        education = int(request.form['education'])

        # Perform predictions using the loaded model
        new_data = [[gender, married, dependents, self_employed, education]]
        prediction = model.predict(new_data)

        # Determine the result message
        if prediction[0] == 1:
            result = "Approved"
        else:
            result = "Not Approved"

        return render_template('index.html', prediction=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
