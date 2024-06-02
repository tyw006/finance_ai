#! /bin/bash
python app/server.py & 
python -m streamlit run app/app.py --server.port=8501 --server.address=0.0.0.0