# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 23:14:57 2020

@author: jeong
"""
import pandas as pd
import numpy as np
import gc, sys, re
sys.path.append('d:/py')
from init import *
from codemap import *
from pandas.tseries.offsets import MonthEnd
"""
1p모델 : joblib.load('model/xgb1p.joblib')
2p이상모델: joblib.load('model/xgb2p.joblib')
만기대상모델: joblib.load('model/xgb_mat.joblib')
"""
#%% rawdata 산출
# pseudo sql code
sql = """
select a.*,b.소분류,b.상품코드
,b.고객id
,b.시도코드
,b.연체발생일
,b.연체해제일
,b.ppd
,c.만기일
,d.대출일자
,d.완제일자
from 연체상담테이블 a
left join 마감테이블 b
 on a.계좌번호=b.계좌번호
 and a.기준년월=b.마감년월
left join 마감테이블 c
 on a.대출번호=c.대출번호
 and a.기준년월 = c.마감월전월
inner join 원장테이블 d
 on a.대출번호=d.대출번호
where a.기준년월 = '{mm}'
 and 상품분류 ='신용'
 """
 mrng = pd.date_range('2019-12','2020-06',freq='MS').strftime('%Y%m').tolist()
 raws = []
 for x in mrng:
     s = tic()
     raws.append(pd.read_sql(sql.format(mm=x),conn))
     print(x,toc(x)) # 한달치 약 2분소요
rx = re.compile(r'[가-힣a-zA-Z]')

def months_between(bg,ed):
    '''bg,ed는 date series'''
    mob = (ed.dt.year - bg.dt.year)*12 + (ed.dt.month - bg.dt.month)
    return mob

#%% 전처리후 데이터 생성
for i,df in enumerate(raws):
    df1 = kornm(df)
    droplist = df1.loc[:,'처리조직':].columns
    df1 = df.drop(droplist,axis=1)
    df1 = df1.replace({'통화유형코드':code1,
                       '통화구분코드':code2,
                       '수신자구분코드':code3,
                       '입금자구분코드':code4,
                       '연체상담통화대분류':code5,
                       '연체상담통화중분류':code6,
                       '연체상담통화소분류':code7})
    datt = df1.초기연체상담일자.astype(str)+' '+df1.상담시각
    datt = pd.to_datetime(datt, format='%Y-%m-%d %H%M%S')
    df1['datt'] = datt
    df1.만기일 = pd.to_datetime(df1.만기일,errors='coerce')
    df1.연체발생일 = pd.to_datetime(df1.연체발생일, errors='coerce')
    df1.연체해제일 = pd.to_datetime(df1.연체해제일, errors='coerce')
    # mob : 연체발생일 - 기준년월
    dat = pd.to_datetime(df1.기준년월,format='%Y%m')
    df1['mob'] = (dat.dt.year-df1.연체발생일.dt.year)*12+(dat.dt.month-df1.연체발생일.dt.month)
    # 기준년월: dat
    df1['만기까지mos'] = months_between(dat,df1.만기일)
    df1.완제일자 = pd.to_datetime(df1.완제일자,errors='coerce')
    # '연체발생일'이 NaT이고 '만기까지mos' <=1이면 만기대상건, 아니면 제외할 것
    df2 = df1.loc[~(df1.연체발생일.isna() & (df1.만기까지mos >1))].copy()
    # 완제일자가 기준년월 이전인 건 제거
    dat1 = pd.to_datetime(df2.기준년월,format='%Y%m')
    df2 = df2.loc[(df2.완제일자 >=dat1)|(df2.완제일자.isna())].copy()
    # '연체해제일'이 기준년월 이전이고 '만기까지mos' >1인 건 제외할 것
    dat2 = pd.to_datetime(df2.기준년월, format='%Y%m')
    df2 = df2.loc[~((df2.연체해제일<dat2) & (df2.만기까지mos >1))].copy()
    # 관리구분
    dat3 = pd.to_datetime(df2.기준년월,format='%Y%m')
    df2.loc[df2.mob.isin([0,-1]),'관리구분'] = '1p대상'
    df2.loc[df2.mob==1,'관리구분'] = '2p대상'
    df2.loc[df2.mob>=2,'관리구분'] = '3p대상'
    df2.loc[df2.연체발생일.isna(),'관리구분'] = '만기대상'
    df2.loc[(df2.만기까지mos<=1),'관리구분'] = '만기대상'
    # target
    dat4 = pd.to_datetime(df2.기준년월,format='%Y%m')
    dat4e = dat4 + MonthEnd(1)
    df2.loc[df2.연체해제일>=dat4,'target'] = '해결'
    df2..loc[(df2.관리구분=='만기대상') & (df2.연체발생일.isna()|(df2.연체해제일<=dat4e)),'target'] = '해결'
    df2['상담시각'] = df2['상담시각'].astype(int)
    cols = ['기준년월','대출번호','대출일자','완제일자','고객id','초기연체상담일','상담시각','연체발생일','연체해제일',
            '만기일','통화유형코드','통화내용','수신자구분코드','상품코드','시도코드','mob','만기까지mos',
            '관리구분','target']
    df2 = df2[cols]
    # 전처리 : 통화내용 없거나 문장부호(.,+/등)만 있는 데이터 제거후 저장
    idx = [0 if str(t) == 'None' or len(rx.findall(t))==0 else 1 for t in df2.통화내용]
    df2['유효text여부'] = idx
    df2 = df2.drop_duplicates()
    df2.to_pickle('data/raw'+str(mrng[i])+'.pkl')
    print(str(mrng([i])),'done')
    gc.collect()
    
#%% 토크나이징,벡터라이징(feature),예측모델링 : raw --> modf
# 1.데이터준비:
#    관리구분별로 통화유형코드가 모형화 대상코드인 건만 남김
#    대출번호별 마지막 값을 가져오는 등의 판단은 여러가지 상황을 모두 고려하기 어려우므로
#    그냥 대출번호별로 모형화 대상코드의 전화통화내용을 벡터화한 후 평균내어 예측모형 만들기
# 2.벡터화 (TfIdf, Doc2Vec)    
# 3.벡터 평균
# 4.예측 모델링 

#%% 토크나이저, 벡터화 모델 로딩
from gensim.models.doc2vec import Doc2Vec
import seaborn as sns
# 토크나이저 모델
# 약어, 은어 사용이 많으므로 비지도학습 접근법을 이용하는 soynlp를 이용하여 토크나이징 모델을 만들었음
# 같은 학습 데이터를 이용하여 tfidf 및 doc2vec 모델 만듬 
tok = pd.read_pickle('model/soy_tok_model.pkl')    
# 벡터화 모델
tfmod = pd.read_pickle('model/tfidf_vect.pkl') 
dvmod = Doc2Vec.load('model/soy_d2v.model') 

# 데이터 준비
raw = pd.read_pickle('data/raw201912.pkl')
raw = raw.loc[raw.유효text여부==1]
raw = raw.loc[raw.통화유형코드.isin(['기타정보','독촉제외','입금약속미정','입금약속확정','재안내'])]
raw = raw[['기준년월','대출번호','관리구분','통화유형코드','통화내용','target']]
raw.reset_index(drop=True,inplace=True)

# 벡터화
# tfidf
tvec = [' '.join(tok(t)) for t in raw.통화내용]
tvec = tfmod.transform(tvec).toarrary()
tvec = pd.DataFrame(tvec)
tvec.columns = ['tf'+str(i+1) for i in range(tvec.shape[1])]
tvec = tvec.reset_index(drop=True)

# doc2vec
dvec = [dvmod.infer_vector(tok(t)) for t in raw.전화통화내용]
dvec = pd.DataFrame(dvec)
dvec.columns = ['dv'+str(i+1) for i in range(dvec.shape[1])]
dvec = dvec.reset_index(drop=True)

# 예측모델용 데이터: 기준년월, 관리구분, 대출번호, target + 벡터
modf = pd.concat([raw[['기준년월','관리구분','대출번호','target']],
                  tvec,dvec], axis=1)
modf = modf.groupby(['기준년월','관리구분','대출번호','target']).mean().reset_index()

#%% 데이터준비-예측모델용 데이터: raw-->modf batch
import glob, re
from sklearn.decomposition import PCA
rawfl = glob.glob('data/raw*.plk')

for f in rawfl:
    # 데이터 준비
    raw = pd.read_pickle('data/raw201912.pkl')
    raw = raw.loc[raw.유효text여부==1]
    raw = raw.loc[raw.통화유형코드.isin(['기타정보','독촉제외','입금약속미정','입금약속확정','재안내'])]
    raw = raw[['기준년월','대출번호','관리구분','통화유형코드','통화내용','target']]
    raw.reset_index(drop=True,inplace=True)
    
    # 벡터화
    # tfidf
    tvec = [' '.join(tok(t)) for t in raw.통화내용]
    tvec = tfmod.transform(tvec).toarrary()
    tvec = pd.DataFrame(tvec)
    tvec.columns = ['tf'+str(i+1) for i in range(tvec.shape[1])]
    tvec = tvec.reset_index(drop=True)
    
    # doc2vec
    dvec = [dvmod.infer_vector(tok(t)) for t in raw.전화통화내용]
    dvec = pd.DataFrame(dvec)
    dvec.columns = ['dv'+str(i+1) for i in range(dvec.shape[1])]
    dvec = dvec.reset_index(drop=True)
    
    modf = pd.concat([raw[['기준년월','관리구분','대출번호','target']],
                      tvec,dvec], axis=1)
    mv = modf.drop('통화내용',axis=1).groupby(['기준년월','관리구분','대출번호','target']).mean().reset_index()
    mt = modf.groupby('대출번호')['통화내용'].sum().reset_index()
    modf = mv.merge(mt,how='inner',on='대출번호').reset_index(drop=True)
    cols = modf.columns.tolist()
    cols1 = cols[:4]+cols[-1:]+cols[4:-1]
    modf = modf[cols1]
    # 저장
    ym = re.findall('\\d{6}',f)[0]
    modf.to_pickle('data/modf'+str(ym)+'.pkl')
    print(ym,'done')
    
#%% 예측모델링 절차 
# 1회차 대상, 2~3회차 대상, 만기대상으로 분리해서 진행
# `19.12~`20.6월까지의 modf 결합 
# `19.12~`20.5 (train:eval= 7:3), test(`20.6)   
# best parameter 적용 및 eval 넣어서 early stopping 적용 ==> 최종모형
# test 데이터로 퍼포먼스 검증 

#%% 라이브러리 로딩
import pandas as pd
import numpy as np
import time
import sklearn.model_selection import train_test_split
import sklearn.utils import resample
import xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             balanced_accuracy_score, f1_score, roc_auc_score)    
#%% metrics 함수 
def metrics(y_true,y_pred,y_score):
    acc = accuracy_score(y_true,y_pred)
    bal_acc = balanced_accuracy_score(y_true,y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true,y_pred)
    auc = roc_auc_score(y_true, y_score)
    print('''
          accuracy: {:.4f}
          balanced accuracy: {:.4f}
          precision: {:.4f}
          recall: {:.4f}
          f1 : {:.4f}
          auc: {:.4f}
          '''.format(acc,bal_ac,prec,recall,f1,auc))
#%% 1p대상 train
# 데이터 결합 
mm = [201912]+list(range(202001,202007))
mm = [str(m) for m in mm]
md1 = []
for m in mm:
    a = pd.read_pickle('data/modf'+m+'.pkl')
    a = a.loc[a.관리구분=='1p대상']
    md1.append(a)
md1 = pd.concat(md1)

# train/eval/test 분리 
trev = md1.loc[md1.기준년월!='202006']
te = md1.loc[md1.기준년월=='202006']
# pc 성능문제로 train_size=0.3
train_d, eval_d = train_test_split(trev, train_size=0.3, random_state=78)
x_train, y_train = train_d.iloc[:,5:], np.where(train_d.target=='해결',1,0).tolist()
x_eval, y_eval = eval_d.iloc[:,5:], np.where(eval_d.target=='해결',1,0).tolist()
x_test, y_test = te.iloc[:,5:], np.where(te['target']=='해결',1,0).tolist()

s = time.time()
xgb = XGBClassifier()
xgb_param_grid = {
        'n_estimators': [600,400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth' : [4,6,8],
        'scale_pos_weight': [0.3],
        'colsample_bytree': [0.8]
        }
xgb_grid = GridSearchCV(xgb, param_grid = xgb_param_grid,
                        scoring = 'balanced_accuracy',n_jobs=-1,verbose=1)
xgb_grid.fit(x_train,y_train)
print('최고 평균 정확도: {:.4f}'.format(xgb_grid.best_score_))
print('최고 파라미터:',xgb_grid.best_params_)
f = time.time() - s
time.strftime('%H:%M:%S',time.gmtime(f))
# 최종모델 fitting (early stopping 적용)
evals = [(x_eval, y_eval)]
xgb1p = XGBClassifier(colsample_bytree=0.8, n_estimators=400,
                      learning_rate=0.05, max_depth=4, scale_pos_weight=0.3,
                      n_jobs=-1)
xgb1p.fit(x_train, y_train, early_stopping_rounds = 100, eval_metric='logloss',
          eval_set = evals, verbose=1)
# train set
tr_pred = xgb1p.predict(x_train)
tr_prob = xgb1p.predict_proba(x_train)[:,1]
metrics(y_train, tr_pred, tr_prob)
# feature importance
fig, ax = plt.subplots(figsize=(10,10))
plot_importance(xgb1p, ax=ax, max_num_features=30)
plt.show()
# 모델 저장
joblib.dump(xgb1p,'model/xgb1p.joblib')
xgb1p = joblib.load('model/xgb1p.joblib')
#%% 2p이상 train
# (생략)
#%% 만기대상 train
# (생략)