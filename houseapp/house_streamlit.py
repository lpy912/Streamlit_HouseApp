import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Klang Valley House Price Prediction App
This app predicts the price of your ideal house in Klang Valley!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    squarefeet = st.sidebar.slider('Squarefeet',1,10000,1000)
    bedroom = st.sidebar.slider('Bedroom number', 1, 10, 3)
    bathroom = st.sidebar.slider('Bathroom number', 1, 10, 2)
    title = st.sidebar.radio('Ideal entitlement of the property',('Freehold', 'Leasehold'))
    hsetype= st.sidebar.radio('House Type', ('Houses','Apartments'))
    bumi= st.sidebar.radio('Other Information', ('Non Bumi Lot','Bumi Lot'))
    user = {'Squarefeet': squarefeet,
            'Bedroom_Number': bedroom,
            'Bathroom_Number': bathroom,
            'Title': title,
            'Type':hsetype,
            'Oth_Info': bumi}
    features = pd.DataFrame(user, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

data=pd.read_csv("cleaned_houseprice.csv")
land=(data['Type']=='Houses') | (data['Type']=='Apartments')
data=data[land==True]
#Remove house price<100000
price=data['Price']>100000
data=data[price==True]
Q1_P = data.Price.quantile(0.25)
Q3_P = data.Price.quantile(0.75)
IQR_P = Q3_P - Q1_P
Outlier_P=(data['Price'] < (Q1_P - 1.5 * IQR_P)) |(data['Price'] > (Q3_P + 1.5 * IQR_P))
data2=data[Outlier_P==False]
Q1 = data.SquareFeet.quantile(0.25)
Q3 = data.SquareFeet.quantile(0.75)
IQR = Q3 - Q1
Outlier=(data['SquareFeet'] < (Q1 - 1.5 * IQR)) |(data['SquareFeet'] > (Q3 + 1.5 * IQR))
data2=data2[Outlier==False]
data2['Title']=data2['Title'].astype('str').replace({"Freehold":1,"Leasehold":0})
data2['Type']=data2['Type'].astype('str').replace({"Houses":1,"Apartments":0})
data2['Oth_Info']=data2['Oth_Info'].astype('str').replace({"Non Bumi Lot":1,"Bumi Lot":0,"Malay Reserved":0})
data2['Bedroom']=data2['Bedroom'].fillna(data['Bedroom'].median())
data2['Bathroom']=data2['Bathroom'].fillna(data['Bathroom'].median())
data2['SquareFeet']=data2['SquareFeet'].fillna(data['SquareFeet'].median())
y=data2['Price']
X=data2.loc[:,['SquareFeet','Bedroom','Bathroom','Title','Type','Oth_Info']]

reg3 = RandomForestRegressor(random_state=1)
reg3.fit(X, y)

df['Title']=df['Title'].astype('str').replace({"Freehold":1,"Leasehold":0})
df['Type']=df['Type'].astype('str').replace({"Houses":1,"Apartments":0})
df['Oth_Info']=df['Oth_Info'].astype('str').replace({"Non Bumi Lot":1,"Bumi Lot":0})


ynew = reg3.predict(df)

st.subheader('Your ideal housetype is estimated to be')
st.write("RM "+"{:.0f}".format(ynew[0]))