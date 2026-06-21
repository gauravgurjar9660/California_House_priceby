
import streamlit as st
import pandas as pd
import numpy as np
import joblib 
data=pd.read_csv("cali-data.csv")
model=joblib.load("model3.pkl")
pipeline=joblib.load("pipline3.pkl")


try:
    
    acc_percentage = joblib.load("accuracy.pkl")  
except:
    acc_percentage = "83.70%"

try:
    # Isme aapka Mean RMSE ($38,903) saved hai
    acc_data = joblib.load("accuracy1.pkl")
    # Agar acc_data dictionary hai toh usse mean_rmse nikalen, nahi toh direct format karein
    if isinstance(acc_data, dict):
        real_rmse = f"${acc_data.get('final_rmse', 38903):,.0f}"
    else:
        real_rmse = "$38,903"
except:
    real_rmse = "$38,903" # Default real-time value agar file na mile

# 2. Sidebar mein dono metrics ko ek sath set karein
with st.sidebar:
    st.header(" Model Performance")
    
    # Do columns banayein taaki dono metrics aamne-saamne mast dikhein
    col_acc, col_rmse = st.columns(2)
    
    with col_acc:
        st.metric(label="R2 Accuracy", value=acc_percentage)
        
    with col_rmse:
        st.metric(label="Real RMSE", value=real_rmse)
        
    # Accuracy ke hisab se progress bar ko set karna
    try:
        progress_val = int(float(acc_percentage.replace("%", "")))
        st.progress(progress_val)
    except:
        st.progress(84)
        
    st.caption("Training based on California Housing Dataset")
st.title("House Cost Pridiction ")
st.write("This app is used for achive the correct amount of any house at a defiend location")
st.image("https://images.unsplash.com/photo-1570129477492-45c003edd2be?w=500", width=300)

st.header("Enter the details below ")

col1,col2 =st.columns(2)
with col1:
    ocean=data['ocean_proximity'].unique().tolist()
    ocean_proximity=st.selectbox("Enter location",ocean)
    latitude=st.number_input("latitude",max_value=42.0,min_value=30.0)
    longitude=st.number_input("longitude",min_value=-124.35,max_value=-114.31)
    housing_median_age=st.number_input("Enter house age",max_value=70,min_value=1)
with col2:
    total_rooms=st.number_input("Total Rooms",value=500)
    total_bedrooms=st.number_input("Total Bedrooms",value=700)
    population=st.number_input("Total Population in area",value=500)
    households=st.number_input("Enter number of Households",value=600)
median_income=st.slider("Enter median income",0.0,15.0,7.2574)

if st.button("SUBMIT"):
    input_dict={
    "longitude":[longitude],"latitude":[latitude],
        "housing_median_age":[housing_median_age],"total_rooms":[total_rooms],"total_bedrooms":[total_bedrooms],
        "population":[population],"households":[households],"median_income":[median_income],"ocean_proximity":[ocean_proximity]}
    df=pd.DataFrame(input_dict)
    df["per_room_household"]=df['total_rooms']/df['households']
    df["population_in_household"]=df["population"]/df["households"]
    
    # 1. distance between Los Anglish and house (distance)
    df["dist_to_LA"] = np.sqrt((df["latitude"] - 34.05)**2 + (df["longitude"] - (-118.24))**2)
    
    # 2. distance between San Francisco and house (distance)
    df["dist_to_SF"] = np.sqrt((df["latitude"] - 37.77)**2 + (df["longitude"] - (-122.41))**2)
    column_order = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms', 
        'total_bedrooms', 'population', 'households', 'median_income', 
        'per_room_household', 'population_in_household', 'dist_to_LA', 'dist_to_SF',
        'ocean_proximity'
    ]
    df = df[column_order]
    try:
        transform_data=pipeline.transform(df)
        log_prediction=model.predict(transform_data)
        prediction = np.expm1(log_prediction)[0]
        st.success(f"The house price is : ${prediction:,.2f}")
        st.subheader("Comparison between your house price to other")
        st.write(" Visualizing by bar chart")
        try:
            averg= data.groupby("ocean_proximity")["median_house_value"].mean().reset_index()
            predicts=pd.DataFrame({"ocean_proximity":['your prediction'],"median_house_value":[prediction]})
            tl=pd.concat([averg,predicts],ignore_index=True)
            st.bar_chart(data=tl,x='ocean_proximity',y='median_house_value',use_container_width=True)
        except Exception as e:
            st.error(f"The Exception is : {e}")

        
    except Exception as e:
        st.error(f"The error occured is {e}")
    

    
    


    

