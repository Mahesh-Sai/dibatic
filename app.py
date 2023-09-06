from flask import Flask, render_template, request

import joblib

app = Flask(__name__)

@app.route('/')
def base():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the model
    model = joblib.load('diabetic_80.pkl')
    
    try:
        # Collect input values from the form
        preg = int(request.form.get('preg'))
        plas = int(request.form.get('plas'))
        pres = int(request.form.get('pres'))
        skin = int(request.form.get('skin'))
        test = int(request.form.get('test'))
        mass = int(request.form.get('mass'))
        pedi = int(request.form.get('pedi'))
        age = int(request.form.get('age'))

        # Make predictions
        input_data = [[preg, plas, pres, skin, test, mass, pedi, age]]
        output = model.predict(input_data)

        # Print the input data and prediction to the console
        print(f"Input Data: preg={preg}, plas={plas}, pres={pres}, skin={skin}, test={test}, mass={mass}, pedi={pedi}, age={age}")
        print(output)
        if output == 0:
            data = 'Person is non-diabetic'
        else:
            data = 'Person is diabetic'
    
    except ValueError as e:
        print(f"Error: {e}")
    
    return render_template('predict.html',data=data)

if __name__ == "__main__":
    app.run(debug=True)