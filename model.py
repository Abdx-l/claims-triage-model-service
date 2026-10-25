import random,json,joblib,numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,cross_val_score
FEATURES=["amount","days_since_start","injury","description_length"]
def generate(count=20000,seed=42):
 rng=random.Random(seed);x=[];y=[]
 for _ in range(count):
  amount=rng.uniform(100,100000);days=rng.randrange(1,1000);injury=rng.randrange(2);length=rng.randrange(20,500)
  label="high" if injury and amount>30000 else "medium" if amount>10000 or days<30 else "low"
  if rng.random()<.05:label=rng.choice(["low","medium","high"])
  x.append([amount,days,injury,length]);y.append(label)
 return np.array(x),np.array(y)
def features(body):
 values=[float(body[k]) for k in FEATURES]
 if not all(np.isfinite(values)) or values[0]<0 or values[1]<0 or values[2] not in (0,1) or values[3]<0:raise ValueError("invalid claim features")
 return values
def train():
 x,y=generate();models={"logistic":make_pipeline(StandardScaler(),LogisticRegression(class_weight="balanced",max_iter=1000)),"forest":RandomForestClassifier(n_estimators=150,class_weight="balanced",random_state=42,n_jobs=-1)}
 scores={name:float(cross_val_score(model,x,y,cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring="f1_macro").mean()) for name,model in models.items()}
 print(json.dumps(scores,indent=2));name=max(scores,key=scores.get);model=models[name].fit(x,y)
 importance=model.feature_importances_ if name=="forest" else np.abs(model[-1].coef_).mean(axis=0)
 joblib.dump({"model":model,"importance":importance,"scores":scores},"triage.joblib")
if __name__=="__main__":train()
