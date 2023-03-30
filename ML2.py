# Complete multiple regressions methods and their ensemble versions
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd 
#----------------------------------------------------------------
#using airlines dataset that follows the below requirements:
# Numeric Target 
# more than one nominal features 
# more than thousand examples
def dataHandling():
    dia = datasets.fetch_openml(data_id = 42721)
    # transforming nominal 
    ct = ColumnTransformer([("encoder",OneHotEncoder(sparse=False),[5,6,7])],remainder="passthrough")
    new_dia = ct.fit_transform(dia.data)
    lis_dia = pd.DataFrame(new_dia,columns = ct.get_feature_names_out(), index = lis_dia.index)
    return new_dia
        
#--------------------------------------------------------------------
def main():
    dia = dataHandling()
    print(type(dia))
    #new_dia = transformNom(dia)
    #print(dia.data)   
if __name__ == "__main__":
    main()
      
