import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.header("Home Page of BridgeStats Project")
st.subheader("To begin, click on one of the links on the left sidebar (scrollable).")
st.subheader("Access to this project is by invitation only.")
st.subheader("This project is in the proof of concept stage. Statistics should be considered provisional, perhaps unreliable.")
st.subheader("Data consists of only ACBL pair game data having hand records. No team data. A significant amount of inconsistent data has been removed.")
st.subheader("Statistics are using ACBL tournament pair results starting with 2015 and ACBL club pair results starting with 2019.")
st.subheader("A Connection Error is usually due to a lack of memory on the server. Try either reloading the webpage immediately or come back later.")
st.subheader("An Error 410 requires that you clear your browser's cache before proceeding.")
st.caption("Project lead is Robert Salita research@AiPolice.org. Code written in Python. UI written in Streamlit. Self-hosted using Cloudflare Tunnel.")
#st.sidebar.header("Home")
