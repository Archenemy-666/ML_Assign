# Complete multiple regressions methods and their ensemble versions
from sklearn import datasets



#----------------------------------------------------------------
#using airlines dataset that follows the below requirements:
# Numeric Target 
# more than one nominal features 
# more than thousand examples
def importingData():
    dia = datasets.fetch_openml(data_id = 42721)
    print(dia.data.info())    

#--------------------------------------------------------------------
def main():
    dia = importingData()
   
if __name__ == "__main__":
    main()
      
