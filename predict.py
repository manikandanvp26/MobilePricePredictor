"""Simple command-line prediction for a dataset-listed mobile."""
import argparse, joblib, pandas as pd
import model_core as core

def main():
    d=pd.read_csv(core.DATA_PATH)
    p=argparse.ArgumentParser(description='Predict a mobile phone price from the trained dataset model')
    p.add_argument('--brand',default=d.brand.iloc[0]); p.add_argument('--model',default=d[d.brand==d.brand.iloc[0]].model.iloc[0])
    for f in core.NUMERIC: p.add_argument('--'+f.replace('_','-'),type=float,default=float(d[f].median()))
    a=p.parse_args(); specs={'brand':a.brand,'model':a.model,**{f:getattr(a,f) for f in core.NUMERIC}}
    b=joblib.load(core.MODEL_PATH); price=float(b['model'].predict(pd.DataFrame([specs],columns=core.FEATURES))[0]); u=float(b.get('uncertainty_abs',price*.15))
    print(f"Phone: {a.brand} {a.model}\nPredicted price: ₹{price:,.0f}\nEstimated range: ₹{max(0,price-u):,.0f} – ₹{price+u:,.0f}")
if __name__=='__main__':main()
