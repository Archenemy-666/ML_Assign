# Complete multiple regressions methods and their ensemble versions
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.svm import SVR
from scipy.stats import ttest_rel

#----------------------------------------------------------------
#using airlines dataset that follows the below requirements:
# Numeric Target 
# more than one nominal features 
# more than thousand examples
def dataHandling():
    dia = datasets.fetch_openml(data_id = 42363)
    # transforming nominal
    # use target of dia -> dia.target for target values and lis_dia for the data
    print(dia.data.info())
    ct = ColumnTransformer([("encoder",OneHotEncoder(sparse=False),[2,3])],remainder="passthrough")
    new_dia = ct.fit_transform(dia.data)
    lis_dia = pd.DataFrame(new_dia,columns = ct.get_feature_names_out(), index = dia.data.index)
    print(lis_dia.info())
    return lis_dia, dia

def LinearRegressionFunction(data,target):
    lr = LinearRegression()
    scores = cross_validate(lr,data,target,cv=10,scoring="neg_root_mean_squared_error")
    rmse = 0 - scores["test_score"]
    mean = rmse.mean()
    
    #bagging 
    bagged_lr = BaggingRegressor(base_estimator=LinearRegression())
    scores_bagged = cross_validate(bagged_lr,data,target,cv=10,scoring="neg_root_mean_squared_error")
    bagged_rmse = 0 - scores_bagged["test_score"]
    bagged_mean = bagged_rmse.mean()
    

    #boosted
    boosted_lr = AdaBoostRegressor(base_estimator = LinearRegression())
    scores_boosted = cross_validate(boosted_lr,data,target,cv=10,scoring="neg_root_mean_squared_error")
    boosted_rmse = 0 - scores_boosted["test_score"]
    boosted_mean = boosted_rmse.mean()

    #VotingRegressor
    vr = VotingRegressor([("lr",LinearRegression()),("svr",SVR())])
    scores_vr = cross_validate(vr,data,target,cv=10,scoring = "neg_root_mean_squared_error")
    vr_rmse = 0 - scores_vr["test_score"]
    vr_mean = vr_rmse.mean()
    
    #statistical significance
    meanVSbagged = ttest_rel(rmse, bagged_rmse)
    meanVSboosted = ttest_rel(rmse, boosted_rmse)
    meanVSvr = ttest_rel(rmse, vr_rmse)    
    print (" Values of Linear Regression:")
    print("mean: ",mean)
    print("bagged mean: ", bagged_mean)
    print("boosted mean: ",boosted_mean)
    print("vr mean: ",vr_mean) 
    print("stat bagged: ",meanVSbagged)
    print("stat boosted: ",meanVSboosted)
    print("stat vr: ",meanVSvr)
    
    return 0

from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection

def DecisionTreeRegressorFunction(data,target):
    dtr = tree.DecisionTreeRegressor()
    parameters = [{"min_samples_leaf":[2,4,6,8,10]}]
    tuned_dtr = model_selection.GridSearchCV(dtr, parameters, cv=5)
    scores = model_selection.cross_validate(tuned_dtr,data,target, cv=10, scoring="neg_root_mean_squared_error")
    rmse = 0-scores["test_score"]
    mean = rmse.mean()
    
    #bagging 
    bagged_dtr = BaggingRegressor(base_estimator=DecisionTreeRegressor())
    scores_bagged = cross_validate(bagged_dtr,data,target,cv=10,scoring="neg_root_mean_squared_error")
    bagged_rmse = 0 - scores_bagged["test_score"]
    bagged_mean = bagged_rmse.mean()
    

    #boosted
    boosted_dtr= AdaBoostRegressor(base_estimator = DecisionTreeRegressor())
    scores_boosted = cross_validate(boosted_dtr,data,target,cv=10,scoring="neg_root_mean_squared_error")
    boosted_rmse = 0 - scores_boosted["test_score"]
    boosted_mean = boosted_rmse.mean()

    #VotingRegressor
    vr = VotingRegressor([("dtr",DecisionTreeRegressor()),("svr",SVR())])
    scores_vr = cross_validate(vr,data,target,cv=10,scoring = "neg_root_mean_squared_error")
    vr_rmse = 0 - scores_vr["test_score"]
    vr_mean = vr_rmse.mean()
    
    #statistical significance
    meanVSbagged = ttest_rel(rmse, bagged_rmse)
    meanVSboosted = ttest_rel(rmse, boosted_rmse)
    meanVSvr = ttest_rel(rmse, vr_rmse)    
    print ("\n Values of Decision Tree Regression:")
    print("mean: ",mean)
    print("bagged mean: ", bagged_mean)
    print("boosted mean: ",boosted_mean)
    print("vr mean: ",vr_mean) 
    print("stat bagged: ",meanVSbagged)
    print("stat boosted: ",meanVSboosted)
    print("stat vr: ",meanVSvr)
 
    return 0

from sklearn.neighbors import KNeighborsRegressor

def KNeighborsFunction(data,target):
    knnr = KNeighborsRegressor()
    parameters = {"n_neighbors":(1,10,1)}
    tuned_knnr = model_selection.GridSearchCV(knnr, parameters, cv=5)
    scores = model_selection.cross_validate(tuned_knnr,data,target, cv=10, scoring="neg_root_mean_squared_error")
    rmse = 0-scores["test_score"]
    mean = rmse.mean() 
    
    #bagging 
    bagged_kn = BaggingRegressor(base_estimator=DecisionTreeRegressor())
    scores_bagged = cross_validate(bagged_kn,data,target,cv=10,scoring="neg_root_mean_squared_error")
    bagged_rmse = 0 - scores_bagged["test_score"]
    bagged_mean = bagged_rmse.mean()

    #boosted
    boosted_kn= AdaBoostRegressor(base_estimator = DecisionTreeRegressor())
    scores_boosted = cross_validate(boosted_kn,data,target,cv=10,scoring="neg_root_mean_squared_error")
    boosted_rmse = 0 - scores_boosted["test_score"]
    boosted_mean = boosted_rmse.mean()

    #VotingRegressor
    vr = VotingRegressor([("dtr",DecisionTreeRegressor()),("svr",SVR())])
    scores_vr = cross_validate(vr,data,target,cv=10,scoring = "neg_root_mean_squared_error")
    vr_rmse = 0 - scores_vr["test_score"]
    vr_mean = vr_rmse.mean()
    
    #statistical significance
    meanVSbagged = ttest_rel(rmse, bagged_rmse)
    meanVSboosted = ttest_rel(rmse, boosted_rmse)
    meanVSvr = ttest_rel(rmse, vr_rmse)    
    print ("\n Values of Kneighbours:")
    print("mean: ",mean)
    print("bagged mean: ", bagged_mean)
    print("boosted mean: ",boosted_mean)
    print("vr mean: ",vr_mean) 
    print("stat bagged: ",meanVSbagged)
    print("stat boosted: ",meanVSboosted)
    print("stat vr: ",meanVSvr)

def SupportVectorMachineFunction(data,target):
    svm = SVR()
    scores = model_selection.cross_validate(svm,data,target, cv=10, scoring="neg_root_mean_squared_error")
    rmse = 0-scores["test_score"]
    mean = rmse.mean()
    
    #bagging 
    bagged_svm = BaggingRegressor(base_estimator=DecisionTreeRegressor())
    scores_bagged = cross_validate(bagged_svm,data,target,cv=10,scoring="neg_root_mean_squared_error")
    bagged_rmse = 0 - scores_bagged["test_score"]
    bagged_mean = bagged_rmse.mean()

    #boosted
    boosted_svm= AdaBoostRegressor(base_estimator = DecisionTreeRegressor())
    scores_boosted = cross_validate(boosted_svm,data,target,cv=10,scoring="neg_root_mean_squared_error")
    boosted_rmse = 0 - scores_boosted["test_score"]
    boosted_mean = boosted_rmse.mean()

    #VotingRegressor
    vr = VotingRegressor([("dtr",DecisionTreeRegressor()),("svr",SVR())])
    scores_vr = cross_validate(vr,data,target,cv=10,scoring = "neg_root_mean_squared_error")
    vr_rmse = 0 - scores_vr["test_score"]
    vr_mean = vr_rmse.mean()
    
    #statistical significance
    meanVSbagged = ttest_rel(rmse, bagged_rmse)
    meanVSboosted = ttest_rel(rmse, boosted_rmse)
    meanVSvr = ttest_rel(rmse, vr_rmse)    
    
    print ("\n Values of SVM:")
    print("mean: ",mean)
    print("bagged mean: ", bagged_mean)
    print("boosted mean: ",boosted_mean)
    print("vr mean: ",vr_mean) 
    print("stat bagged: ",meanVSbagged)
    print("stat boosted: ",meanVSboosted)
    print("stat vr: ",meanVSvr)

 

#--------------------------------------------------------------------
def main():
    #dia = dataHandling()
    
    lis_dia , dia = dataHandling()
    #print(type(dia), type(lis_dia))
    LinearRegressionFunction(lis_dia,dia.target)
    DecisionTreeRegressorFunction(lis_dia,dia.target)
    KNeighborsFunction(lis_dia,dia.target)
    SupportVectorMachineFunction(lis_dia,dia.target) 
    #print("stat bagged: ",meanVSbagged)
    #print("stat boosted: ",meanVSboosted)
    #print("stat vr: ",meanVSvr)
    #new_dia = transformNom(dia)
    #print(dia.data)   
if __name__ == "__main__":
    main()
      
