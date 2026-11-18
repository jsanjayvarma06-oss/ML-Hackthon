"""Evaluator UI (Laptop 2). OWNER: Samarth.

Run:  streamlit run ui/app.py
Talks to the API only. Never reads data files directly.
"""
import requests
import streamlit as st

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap route reconstruction")

api_url = st.sidebar.text_input("API URL (Laptop 1)", "http://127.0.0.1:8000")

try:
    trip_ids = requests.get(f"{api_url}/trips", timeout=10).json()["trips"]
except Exception as e:
    st.error(f"Cannot reach the API at {api_url}: {e}")
    st.stop()

trip_id = st.selectbox("Evaluation trip", trip_ids)
if st.button("Reconstruct route"):
    resp = requests.post(f"{api_url}/predict", json={"trip_id": trip_id}, timeout=300)
    if resp.status_code != 200:
        st.error(f"API error {resp.status_code}: {resp.text}")
        st.stop()
    result = resp.json()
    st.json(result["metrics"])
    # TODO(Samarth): folium map with GPS points + predicted_path, legend and LayerControl
