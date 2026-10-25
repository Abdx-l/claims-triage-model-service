from functools import lru_cache
import joblib
from flask import Flask,request,jsonify
from model import features,FEATURES
app=Flask(__name__)
@lru_cache
def artifact():return joblib.load("triage.joblib")
@app.post("/triage")
def triage():
 try:x=features(request.get_json())
 except (KeyError,ValueError,TypeError):return jsonify(error="invalid claim features"),400
 a=artifact();probabilities=a["model"].predict_proba([x])[0];i=int(probabilities.argmax())
 return jsonify(label=str(a["model"].classes_[i]),probability=float(probabilities[i]),global_feature_importance=[{"feature":FEATURES[j],"importance":float(a["importance"][j])} for j in a["importance"].argsort()[-3:][::-1]])
