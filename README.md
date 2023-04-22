# BridgeStats

This project contains source code for an ACBL bridge statistics website. 

The website server requires at least 64GB of memory.

Website is written in Streamlit. Dependencies are in requirements.txt.

To install:

`pip install -r requirements.txt`

To run:

`streamlit run Home.py`

You can host the website locally and use cloudflare tunnels to forward streamlit port 8501 for public access.

Data is only available, from this project's author, for those who sign ACBL's NDA or register for ACBL's API. The data is primarily oriented for consumption by a machine learning model as opposed to human readability. The data is created by both scraping ACBL's website and using their APIs. The data creation process is extremely resource intensive and requires at least 128GB multi-core systems. Source code for building the data is not yet available.

## Related Projects
https://github.com/dds-bridge/dds
https://github.com/gpaulissen/bml and https://github.com/gpaulissen/bridge-systems and https://github.com/jdh8/bml
https://github.com/gpaulissen/bridge-systems
https://github.com/dominicprice/endplay
https://github.com/anntzer/redeal
https://github.com/liu1000/bidly
https://github.com/kdblocher/bridge
https://github.com/ContractBridge/pbn-files
https://github.com/jfklorenz/Bridge-Package
https://github.com/jfklorenz/Card-Deck-Package
https://github.com/jfklorenz/Bridge-Scoring-Package
https://github.com/kdblocher/bridge
https://github.com/ContractBridge/pbnj
https://github.com/Kungsgeten/bml

## Related Documents
https://www.tistis.nl/pbn/
https://github.com/jfklorenz/Bridge-Documents
http://home.claranet.nl/users/veugent/pbn/pbn_v20.txt
