from fastapi import FastAPI
import uvicorn
from joblib import Parallel, delayed
import joblib

file = "../Model/gender_prediction_pipeline_3_gender.pkl"

app = FastAPI(debug=True)

@app.post('/predict-gender')
def predict_gender(input: str):
    model = joblib.load(file)
    prediction = model.predict_proba([input.lower()]).tolist()
    kamus = dict(enumerate(model.classes_,0))
    high = max(prediction[0])

    return {"result": kamus[prediction[0].index(high)], "probability": high}

if __name__ == '__main__':
    uvicorn.run(app)