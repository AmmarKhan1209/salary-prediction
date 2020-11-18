# %%
import numpy as np
import pickle
from flask import Flask,request,jsonify,render_template

app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))

# %%
@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict",methods = ["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    prediction = model.predict(final_features)
    output = int(np.round(prediction[0][0]))
    
    return render_template("index.html",prediction_text=f"Salary should be Rs. {output}")

# %%
if __name__ == "__main__":
    app.run(debug=True,use_reloader = False)

# %%


# %%
