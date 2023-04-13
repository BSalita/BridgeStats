# BridgeStats

Source code for an ACBL bridge statistics website. Data is only available for those who sign ACBL's NDA or register for ACBL's API. The website requires at least 64GB of memory.

Website is written in Streamlit. Dependencies are in requirements.txt.

pip install -r requirements.txt

To run:

streamlit run Home.py

You can host the website locally and use cloudflare tunnels to forward streamlit port 8501 for public access.
