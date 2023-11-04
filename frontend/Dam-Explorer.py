import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="oDam",
    page_icon="🌊",
    initial_sidebar_state="expanded"
)

st.image("frontend/images/oDam_logo.png")

## dummy data ##
capacity = 88
df = pd.read_csv('frontend/dams.csv', header=0, low_memory=False)
###


option = st.selectbox('Select Dam',(df['name']))
update = st.button("Find")
if update:
    selected_df = df[(df['name'] == option)]
    st.map(selected_df, zoom=2, color='#008000')

    short_tab, long_tab =  st.tabs(["Short-term Forecast", "Long-term Forecast"])

    with short_tab:
        col1, col2 = st.columns(2)

        dummy_data = pd.DataFrame()
        dummy_data["Date"] = np.linspace(0, 10, 100)
        dummy_data["Inflow"] = [np.sin(i) for i in dummy_data["Date"]]
        dummy_data["Outflow"] = [np.cos(i) for i in dummy_data["Date"]]
        dummy_data["Storage"] = [np.tan(i) for i in dummy_data["Date"]]


        with col1:
            st.line_chart(dummy_data, x="Date", y="Inflow")
            st.line_chart(dummy_data, x="Date", y="Storage")

        with col2:
            st.line_chart(dummy_data, x="Date", y=["Inflow", "Outflow"])


    with long_tab:
        st.write("Placeholder")