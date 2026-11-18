import io, json, sqlite3, datetime, os
import joblib, numpy as np, pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
import model_core as core
app=Flask(__name__); app.config['MAX_CONTENT_LENGTH']=8*1024*1024
_state={'bundle':None,'df':None,'models':{}}
def get_df():
    if _state['df'] is None: _state['df']=pd.read_csv(core.DATA_PATH)
    return _state['df']
def get_bundle():
    if _state['bundle'] is None:
        if not core.MODEL_PATH.exists():
            import train; train.main()
        _state['bundle']=joblib.load(core.MODEL_PATH)
    return _state['bundle']
def get_model_bundle(name):
    # Returns a model the user picked on the evaluation page; the deployed model is reused when it matches.
    if name not in core.MODEL_NAMES: raise ValueError('Select a valid ML model.')
    best=get_bundle()
    if name==best['best_model']: return best
    if name not in _state['models']:
        d=get_df().dropna(subset=[core.TARGET])
        fitted=core.make_pipeline(name).fit(d[core.FEATURES],d[core.TARGET].astype(float))
        unc=json.loads(core.METRICS_PATH.read_text())['models'][name].get('uncertainty_abs_90')
        _state['models'][name]={'model':fitted,'best_model':name,'uncertainty_abs':unc}
    return _state['models'][name]
def db():
    con=sqlite3.connect(core.DB_PATH); con.row_factory=sqlite3.Row
    con.execute("CREATE TABLE IF NOT EXISTS wishlist(id INTEGER PRIMARY KEY AUTOINCREMENT, brand TEXT, model TEXT, specs TEXT, created_at TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY AUTOINCREMENT, brand TEXT, model TEXT, specs TEXT, target REAL, active INTEGER DEFAULT 1, triggered INTEGER DEFAULT 0, created_at TEXT)")
    con.commit(); return con
def clean_specs(raw):
    df=get_df(); brands=sorted(df.brand.astype(str).unique()); brand=str(raw.get('brand',brands[0]))
    if brand not in brands: raise ValueError('Select a brand from the dataset.')
    models=sorted(df.loc[df.brand==brand,'model'].astype(str).unique()); model=str(raw.get('model',models[0] if models else ''))
    if model not in models: raise ValueError('Select a model available for this brand.')
    out={'brand':brand,'model':model}
    for f in core.NUMERIC:
        try: v=float(raw.get(f,df[f].median()))
        except (TypeError,ValueError): raise ValueError(core.FEATURE_LABELS[f]+' must be numeric.')
        lo,hi=core.NUMERIC_RANGES[f]
        if not lo<=v<=hi: raise ValueError(f'{core.FEATURE_LABELS[f]} must be between {lo} and {hi}.')
        out[f]=v
    return out
def predict_value(specs,ml_model=None):
    b=get_model_bundle(ml_model) if ml_model else get_bundle(); x=pd.DataFrame([specs],columns=core.FEATURES); p=max(0,float(b['model'].predict(x)[0])); u=float(b.get('uncertainty_abs') or p*.12)
    return {'ml_model':b['best_model'],'ml_label':core.MODEL_LABELS[b['best_model']],'specs':specs,'price':round(p),'price_fmt':f'₹{p:,.0f}','low':round(max(0,p-u)),'high':round(p+u),'low_fmt':f'₹{max(0,p-u):,.0f}','high_fmt':f'₹{p+u:,.0f}','uncertainty':round(u),'category':core.price_category(p)}
def fairness(specs,pred):
    d=get_df(); sub=d[d.brand==specs['brand']]
    if len(sub)==0: sub=d
    # nearest comparable rows using standardized numeric distance
    cols=core.NUMERIC; scale=d[cols].std().replace(0,1).fillna(1)
    dist=((sub[cols]-pd.Series({c:specs[c] for c in cols}))/scale).pow(2).sum(axis=1)
    near=sub.loc[dist.nsmallest(min(8,len(sub))).index,'price_inr']
    median=float(near.median()) if len(near) else pred
    delta=(pred-median)/max(median,1)*100
    score=int(np.clip(100-abs(delta)*1.6,0,100))
    return {'score':score,'label':'Fair' if abs(delta)<=10 else ('Above comparable listings' if delta>0 else 'Below comparable listings'),'comparable_median':round(median),'difference_pct':round(delta,1)}
def future_price(p,years,annual_rate):
    years=float(years); rate=float(annual_rate)
    if not 0<=years<=5 or not 0<=rate<=50: raise ValueError('Forecast years must be 0–5 and annual depreciation 0–50%.')
    return round(p*((1-rate/100)**years))
@app.get('/')
def index(): return render_template('index.html')
@app.get('/api/schema')
def schema():
    d=get_df(); brands=sorted(d.brand.astype(str).unique())
    first=brands[0]; rec=d[d.brand==first].iloc[0]
    return jsonify({'brands':brands,'models':sorted(d[d.brand==first].model.astype(str).unique()),'labels':core.FEATURE_LABELS,'defaults':{'brand':first,'model':str(rec.model),**{f:float(rec[f]) for f in core.NUMERIC}},'model_name':core.MODEL_LABELS[get_bundle()['best_model']]})
@app.get('/api/models')
def models():
    d=get_df(); return jsonify({'phones':d[['brand','model']].drop_duplicates().sort_values(['brand','model']).to_dict('records')})
@app.get('/api/models/<path:brand>')
def models_for_brand(brand):
    d=get_df(); return jsonify({'models':sorted(d[d.brand==brand].model.astype(str).unique().tolist())})
@app.get('/api/phone')
def phone_lookup():
    brand=request.args.get('brand',''); model=request.args.get('model',''); d=get_df(); r=d[(d.brand==brand)&(d.model==model)]
    if r.empty:return jsonify(error='Phone not found in dataset.'),404
    row=r.iloc[0]; return jsonify({'brand':brand,'model':model,**{f:float(row[f]) for f in core.NUMERIC},'actual_price':float(row.price_inr)})
@app.post('/api/predict')
def predict():
    try:
        data=request.get_json(silent=True) or {}; s=clean_specs(data); out=predict_value(s,data.get('ml_model')); out['fairness']=fairness(s,out['price']); return jsonify(out)
    except ValueError as e:return jsonify(error=str(e)),400
@app.post('/api/compare')
def compare():
    try:
        phones=(request.get_json(silent=True) or {}).get('phones',[])
        if not 1<=len(phones)<=4: raise ValueError('Compare 1 to 4 phones.')
        return jsonify({'phones':[predict_value(clean_specs(p)) for p in phones]})
    except ValueError as e:return jsonify(error=str(e)),400
@app.post('/api/whatif')
def whatif():
    try:
        s=clean_specs(request.get_json(silent=True) or {}); base=predict_value(s); rows=[]
        for k in ['ram_gb','storage_gb','battery_mah','main_camera_mp']:
            for v in sorted(set([s[k],max(core.NUMERIC_RANGES[k][0],s[k]*.75),min(core.NUMERIC_RANGES[k][1],s[k]*1.25)])):
                q=dict(s); q[k]=v; rows.append({'feature':core.FEATURE_LABELS[k],'value':v,'price':predict_value(q)['price']})
        return jsonify({'base':base,'scenarios':rows})
    except ValueError as e:return jsonify(error=str(e)),400
@app.post('/api/forecast')
def forecast():
    try:
        s=clean_specs(request.get_json(silent=True) or {}); p=predict_value(s)['price']; years=float(request.json.get('years',1)); rate=float(request.json.get('annual_depreciation_pct',10)); future=future_price(p,years,rate)
        return jsonify({'current_price':p,'future_price':future,'future_fmt':f'₹{future:,.0f}','years':years,'annual_depreciation_pct':rate,'assumption':'Scenario estimate only; not learned from historical market prices.'})
    except (ValueError,TypeError) as e:return jsonify(error=str(e)),400
@app.post('/api/recommend')
def recommend():
    try:
        data=request.get_json(silent=True) or {}; budget=float(data.get('budget',30000)); brand=data.get('brand','All'); minram=float(data.get('min_ram',4)); minstorage=float(data.get('min_storage',64)); d=get_df(); cand=d[(d.price_inr<=budget)&(d.ram_gb>=minram)&(d.storage_gb>=minstorage)]
        if brand!='All':cand=cand[cand.brand==brand]
        if cand.empty:return jsonify({'recommendations':[],'message':'No matching phones. Increase budget or lower the specification filters.'})
        cand=cand.sort_values(['rating','price_inr'],ascending=[False,True]).drop_duplicates(['brand','model']).head(12)
        return jsonify({'recommendations':cand[['brand','model','rating','ram_gb','storage_gb','battery_mah','price_inr']].to_dict('records')})
    except (ValueError,TypeError) as e:return jsonify(error=str(e)),400
@app.get('/api/analytics')
def analytics():
    d=get_df(); return jsonify({'records':len(d),'brands':int(d.brand.nunique()),'min_price':float(d.price_inr.min()),'median_price':float(d.price_inr.median()),'max_price':float(d.price_inr.max()),'brand_counts':d.brand.value_counts().head(12).to_dict(),'price_bands':{'Under ₹10k':int((d.price_inr<10000).sum()),'₹10k–25k':int(((d.price_inr>=10000)&(d.price_inr<25000)).sum()),'₹25k–50k':int(((d.price_inr>=25000)&(d.price_inr<50000)).sum()),'Above ₹50k':int((d.price_inr>=50000).sum())},'average_specs':{f:round(float(d[f].mean()),2) for f in core.NUMERIC}})
@app.get('/api/metrics')
def metrics(): return jsonify(json.loads(core.METRICS_PATH.read_text()))
@app.get('/api/insights')
def insights(): return jsonify(json.loads(core.METRICS_PATH.read_text()).get('feature_importance',[]))
@app.post('/api/explain')
def explain():
    try:
        s=clean_specs(request.get_json(silent=True) or {}); bundle=get_bundle(); d=get_df()
        import shap
        fitted=bundle['model'].regressor_; prep=fitted.named_steps['prep']; estimator=fitted.named_steps['model']
        background=prep.transform(d[core.FEATURES].sample(min(20,len(d)),random_state=42))
        point=prep.transform(pd.DataFrame([s],columns=core.FEATURES))
        if hasattr(background,'toarray'): background=background.toarray()
        if hasattr(point,'toarray'): point=point.toarray()
        feature_names=prep.get_feature_names_out()
        # The estimator learns log(price); invert inside the prediction function so SHAP values are in INR.
        predict_inr=lambda arr: np.expm1(estimator.predict(arr))
        explainer=shap.Explainer(predict_inr,background,feature_names=feature_names)
        explanation=explainer(point,max_evals=2*background.shape[1]+1)
        grouped={f:0.0 for f in core.FEATURES}
        for name,value in zip(feature_names,explanation.values[0]):
            if name.startswith('cats__brand_'): grouped['brand']+=float(value)
            elif name.startswith('cats__model_'): grouped['model']+=float(value)
            elif name.startswith('nums__'):
                key=name.split('__',1)[1]
                if key in grouped: grouped[key]+=float(value)
        base=float(np.array(explanation.base_values).reshape(-1)[0])
        return jsonify({'base_value':round(base),'explanations':[{'feature':core.FEATURE_LABELS[f],'value':str(s[f]),'impact':round(v,2)} for f,v in sorted(grouped.items(),key=lambda x:-abs(x[1]))]})
    except ImportError:return jsonify(error='SHAP is not installed yet. Run: pip install -r requirements.txt'),503
    except Exception as e:return jsonify(error='Could not calculate SHAP explanation: '+str(e)),500
@app.get('/api/wishlist')
def wishlist_get():
    c=db(); rows=[dict(r) for r in c.execute('SELECT * FROM wishlist ORDER BY id DESC')]; c.close();
    for r in rows:r['specs']=json.loads(r['specs'])
    return jsonify({'items':rows})
@app.post('/api/wishlist')
def wishlist_add():
    try:
        s=clean_specs(request.get_json(silent=True) or {}); c=db(); cur=c.execute('INSERT INTO wishlist(brand,model,specs,created_at) VALUES(?,?,?,?)',(s['brand'],s['model'],json.dumps(s),datetime.datetime.now().isoformat())); c.commit(); i=cur.lastrowid;c.close();return jsonify({'id':i,'message':'Added to wishlist.'})
    except ValueError as e:return jsonify(error=str(e)),400
@app.delete('/api/wishlist/<int:item_id>')
def wishlist_delete(item_id):
    c=db(); c.execute('DELETE FROM wishlist WHERE id=?',(item_id,)); c.commit();c.close();return jsonify({'ok':True})
@app.get('/api/alerts')
def alerts_get():
    c=db(); rows=[dict(r) for r in c.execute('SELECT * FROM alerts ORDER BY id DESC')]; c.close()
    for r in rows:r['specs']=json.loads(r['specs'])
    return jsonify({'items':rows})
@app.post('/api/alerts')
def alert_add():
    try:
        data=request.get_json(silent=True) or {}; s=clean_specs(data); target=float(data.get('target',0))
        if target<=0:raise ValueError('Enter a target price above zero.')
        c=db();cur=c.execute('INSERT INTO alerts(brand,model,specs,target,created_at) VALUES(?,?,?,?,?)',(s['brand'],s['model'],json.dumps(s),target,datetime.datetime.now().isoformat()));c.commit();i=cur.lastrowid;c.close();return jsonify({'id':i,'message':'In-app price alert saved.'})
    except ValueError as e:return jsonify(error=str(e)),400
@app.delete('/api/alerts/<int:item_id>')
def alert_delete(item_id):
    c=db();c.execute('DELETE FROM alerts WHERE id=?',(item_id,));c.commit();c.close();return jsonify({'ok':True})
@app.post('/api/alerts/check')
def alerts_check():
    c=db(); rows=c.execute('SELECT * FROM alerts WHERE active=1 AND triggered=0').fetchall(); triggered=[]
    for r in rows:
        s=json.loads(r['specs']); p=predict_value(s)['price']
        if p<=r['target']:
            c.execute('UPDATE alerts SET triggered=1 WHERE id=?',(r['id'],)); triggered.append({'id':r['id'],'brand':r['brand'],'model':r['model'],'predicted':p,'target':r['target']})
    c.commit();c.close();return jsonify({'triggered':triggered,'note':'Alerts compare the current model estimate with your target, not live retailer prices.'})
@app.get('/api/template')
def template():
    from flask import send_from_directory
    return send_from_directory(core.ROOT/'static','phone_template.csv',as_attachment=True)
@app.post('/api/batch')
def batch():
    f=request.files.get('file')
    if not f:return jsonify(error='Choose a CSV or Excel file.'),400
    try:
        df=pd.read_excel(f) if f.filename.lower().endswith(('.xlsx','.xls')) else pd.read_csv(f)
        rows=[]
        for _,r in df.head(2000).iterrows():
            try:
                s=clean_specs(r.to_dict()); p=predict_value(s); rows.append({'brand':s['brand'],'model':s['model'],'predicted':p['price']})
            except Exception as e:rows.append({'error':str(e)})
        return jsonify({'total':len(rows),'rows':rows})
    except Exception as e:return jsonify(error='Unable to read upload: '+str(e)),400
@app.get('/api/report')
def report():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm
        s=clean_specs({k:request.args.get(k) for k in core.FEATURES}); p=predict_value(s); fair=fairness(s,p['price'])
        buf=io.BytesIO(); c=canvas.Canvas(buf,pagesize=A4); w,h=A4; y=h-25*mm
        c.setFont('Helvetica-Bold',18);c.drawString(20*mm,y,'Mobile Price Predictor Report');y-=14*mm
        c.setFont('Helvetica',11)
        for line in [f"Phone: {s['brand']} {s['model']}",f"Predicted price: INR {p['price']:,}",f"Prediction range: INR {p['low']:,} - INR {p['high']:,}",f"Price fairness: {fair['score']}/100 ({fair['label']})",'','Specifications:']+[f'{core.FEATURE_LABELS[k]}: {s[k]}' for k in core.FEATURES[2:]]+['','Note: estimate based on uploaded dataset; not a live retailer quote.']:
            c.drawString(20*mm,y,line[:110]);y-=8*mm
        c.save();buf.seek(0);return send_file(buf,mimetype='application/pdf',as_attachment=True,download_name='mobile_price_report.pdf')
    except Exception as e:return jsonify(error=str(e)),400
@app.errorhandler(413)
def too_large(e):return jsonify(error='Upload is too large (maximum 8 MB).'),413
if __name__=='__main__':app.run(host='0.0.0.0',port=int(os.environ.get('PORT',5000)),debug=False)
