import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
import requests
from streamlit_lottie import st_lottie
import plotly.graph_objects as go
from pathlib import Path


st.set_page_config(page_title="Project Webpage",
                   page_icon=":bar_chart:", layout='wide')


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_coding = load_lottieurl(
    "https://assets1.lottiefiles.com/packages/lf20_49rdyysj.json")


st.title("Bank Statement Data Analysis Project:bar_chart:")

with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)

    with left_column:
        st.header("About the Project")
        st.write(
            "Hi my name is [Eric Kye](https://www.linkedin.com/in/eric-kye/) and I am a second year Statistics and Data Science Student at UC Davis. I am excited to present my first data science related personal project! For this project my main objectives were to get comfortable working with the Python Pandas library, work with data visualizations and graphs, and also implement statistical concepts like linear regression modeling. \nFor the dataset, I thought it would be cool to look at my own bank records to see if I could notice any trends with my spending. I was able to download my bank records as a CSV file and work with that dataset in python. ")
        st.markdown('''

[Github Repo](https://github.com/ekye0512/Personal_Bank_Budgeting)
''')
    with right_column:
        st_lottie(lottie_coding, height=400, key="data")
st.write("---")


df_mint = pd.read_csv(
    '/Users/eric/Documents/Github/Personal_Bank_Budgeting/transactions.csv')


for x in df_mint.index:
    if df_mint.loc[x, 'Category'] == 'Income' or df_mint.loc[x, 'Category'] == 'Paycheck':
        df_mint = df_mint.drop(x)


df_mint = df_mint.loc[::-1]

df_mint['Date'] = pd.to_datetime(df_mint['Date'])
df_mint['Month'] = df_mint['Date'].dt.month

st.header("Graphs :chart_with_upwards_trend:")


fig1 = px.histogram(df_mint, x='Category',
                    title="Distribution of the Types of Transactions")
fig1.update_xaxes(categoryorder='total ascending')
st.write(fig1)


st.markdown('### ')
fig = px.line(df_mint, x='Date', y="Amount",
              title="Money Spent Through Time")
st.write(fig)

fig0 = px.bar(df_mint, x='Month', y='Amount',
              title='Bar Chart of Month vs Money Spent')
st.write(fig0)


fig1 = px.scatter(
    df_mint, x='Month', y='Amount', opacity=0.65,
    trendline='ols', trendline_color_override='red', title="Linear Regression Model for Month vs Amount Spent"
)
fig1['data'][1]['showlegend'] = True
fig1['data'][1]['name'] = 'Regression Line'

st.write(fig1)

st.write("---")

st.header('Overall takeways from the project ')

st.markdown("- There is no clear linear relationship between month and money spent \n - I do most of my purchases at transfers(Zelle or Venmo), fast food, and restaurants\n - Most of the money value is spent on transfers, rent, and shopping\n - Most of my spending is done around winter(During holiday break from school when I go back home)\n - The amount I have spent has increased recently(Rent and Utilities and Groceries is the main reason)\n - Overall looking at the linear regression model, my spending is consistent throughout the year with few big outliers that come(mostly from rent)\n\n What's next?\n - Try to learn how to work with APIS and real time data\n - Try to learn data cloud services\n - Learn machine learning principles to be able to apply models to my future projects \n - Learn SQL and retouch on my python")


with st.container():
    st.write("---")
    st.header("Get In Touch With Me!")
    st.write("##")

    # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
    contact_form = """
    <form action="https://formsubmit.co/ekye0512@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()
