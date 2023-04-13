# BridgeStats

This project contains source code for an ACBL bridge statistics website. Data is only available, from this project's author, for those who sign ACBL's NDA or register for ACBL's API. The data is primarily oriented for consumption by a machine learning model as opposed to human readability. The data is created by both scraping ACBL's website and using their APIs. The data creation process is extremely resource intensive and requires at least 128GB multi-core systems. Source code for building the data is not yet available.

The website requires at least 64GB of memory.

Website is written in Streamlit. Dependencies are in requirements.txt.

pip install -r requirements.txt

To run:

streamlit run Home.py

You can host the website locally and use cloudflare tunnels to forward streamlit port 8501 for public access.
