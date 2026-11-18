"""Evaluator UI (Laptop 2). OWNER: Kushal (handed over from Samarth on 18 Nov, see CLAUDE.md request).

Run:  streamlit run ui/app.py
Talks to the API only. It never reads data files, never sees the reference route,
and shows only numbers the API returns.
"""
import math

import requests
import streamlit as st
from streamlit_folium import st_folium

from map_view import make_result_map  # ui/ is on sys.path when Streamlit runs ui/app.py

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap route reconstruction")
st.caption(
    "Pick an evaluation trip. Laptop 1 matches its noisy GPS points to road segments "
    "(classifier + HMM/Viterbi) and sends back the predicted route and its accuracy."
)


# ---------- API calls ----------

class ApiError(Exception):
    """An error message that is safe to show to the evaluator."""


def _call(method, api_url, path, timeout, **kwargs):
    try:
        resp = requests.request(method, f"{api_url.rstrip('/')}{path}", timeout=timeout, **kwargs)
    except requests.Timeout:
        raise ApiError(f"The API at {api_url} did not answer within {timeout} s.")
    except requests.RequestException:
        raise ApiError(f"Cannot reach the API at {api_url}. Check the IP and that both laptops "
                       "are on the same hotspot.")
    if resp.status_code == 200:
        return resp.json()
    if resp.status_code == 404:
        raise ApiError("Unknown trip id.")
    if resp.status_code == 501:
        raise ApiError("The model pipeline is not ready yet.")
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    raise ApiError(f"API error {resp.status_code}: {detail}")


@st.cache_data(ttl=60, show_spinner=False)
def get_trips(api_url):
    return _call("GET", api_url, "/trips", timeout=10)["trips"]


def get_health(api_url):
    return _call("GET", api_url, "/health", timeout=10)


def predict(api_url, trip_id):
    return _call("POST", api_url, "/predict", timeout=300, json={"trip_id": trip_id})


def pct(x):
    """0.8312 -> '83.1%'; missing or NaN -> 'N/A'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "N/A"
    return f"{x * 100:.1f}%"


# ---------- Sidebar: connection ----------

api_url = st.sidebar.text_input("API URL (Laptop 1)", "http://127.0.0.1:8000").strip()
if st.sidebar.button("Check connection"):
    try:
        st.session_state["health"] = (api_url, get_health(api_url), None)
    except ApiError as e:
        st.session_state["health"] = (api_url, None, str(e))

# Keep the last check on screen across reruns, but only for the URL it was made against.
checked = st.session_state.get("health")
if checked and checked[0] == api_url:
    _, health, err = checked
    if err:
        st.sidebar.error(err)
    else:
        st.sidebar.success(f"API is up. Trips folder: {health.get('eval_dir', '?')}")
        if health.get("model"):
            # Shown only if the API reports it (requested in docs/REQUESTS.md); never typed by hand.
            st.sidebar.info(f"Model on Laptop 1: {health['model']}")

# ---------- Trip choice ----------

try:
    trip_ids = get_trips(api_url)
except ApiError as e:
    st.error(str(e))
    st.stop()

if not trip_ids:
    st.warning("The API returned no trips. Check PATHSNAP_EVAL_DIR on Laptop 1.")
    st.stop()

trip_id = st.selectbox(f"Evaluation trip ({len(trip_ids)} available)", trip_ids)

if st.button("Reconstruct route", type="primary"):
    try:
        with st.spinner("Running the pipeline on Laptop 1..."):
            st.session_state["result"] = predict(api_url, trip_id)
    except ApiError as e:
        st.session_state.pop("result", None)
        st.error(str(e))

# ---------- Result (kept in session_state so map clicks don't erase it) ----------

result = st.session_state.get("result")
if result and result["trip_id"] == trip_id:
    m = result.get("metrics", {})
    timing = result.get("timing_ms", {})
    total = timing.get("total")

    st.subheader(f"Trip {trip_id}: {result.get('n_points', 0):,} GPS points")
    c1, c2, c3 = st.columns(3)
    c4, c5, _ = st.columns(3)
    c1.metric("Top-1 accuracy (final)", pct(m.get("top1_accuracy")),
              help="Share of scored GPS points matched to the correct road segment after HMM/Viterbi.")
    c2.metric("Top-1 accuracy (classifier only)", pct(m.get("top1_accuracy_classifier_only")),
              help="Same, using only the classifier's best candidate per point (no HMM).")
    c3.metric("Route continuity", pct(m.get("route_continuity")),
              help="Share of consecutive predicted segments that are actually connected.")
    c4.metric("Points scored", f"{m.get('n_scored_points', 0):,}",
              help="GPS points that had a reference label. Scoring runs on Laptop 1.")
    c5.metric("Total time", "N/A" if total is None else f"{total:,.0f} ms",
              help="Pipeline time on Laptop 1.")

    st_folium(make_result_map(result), height=600, use_container_width=True, returned_objects=[])

    if timing:
        st.markdown("**Time per step on Laptop 1**")
        st.table([{"step": k, "time (ms)": f"{v:,.1f}"} for k, v in timing.items()])
