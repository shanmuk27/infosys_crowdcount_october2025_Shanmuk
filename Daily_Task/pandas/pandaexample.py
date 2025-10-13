import pandas as pd

data = pd.read_csv(r'C:\Users\pooji\OneDrive\Desktop\croud count\Daily_Task\archive\IRIS.csv')
data_frame=pd.DataFrame(data)
print(data_frame)
print(data.head(10)) #prints first 10
print(data.tail()) #prints last 5
print(data.info()) 

#for removing empty sets
new_emt_rem=data.dropna() #removes the empty sets
print(new_emt_rem)

filling_data=data.fillna(123) #fills the empty set
print(filling_data)

mean_data=data['sepal_length'].mean()
near_fix_sepal_length=data.fillna({'sepal_length':mean_data})
print(near_fix_sepal_length)

'''for date and time format
to correct incorrect 
df['Date'] = pd.to_datetime(df['Date'], format='mixed')
to remove still incorrect one
df.dropna(subset=['Date'], inplace = True)
here inplace = True will change the existing data
'''

#for finding duplicates
data_sepal_length_cleen=data.duplicated()
data_sepal_length_cleen.drop_duplicates(inplace=True)
print(data_sepal_length_cleen.to_string())