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


def get_models(api_url):
    """Model names the API offers (first = default). Empty list on an older API."""
    try:
        return _call("GET", api_url, "/models", timeout=10).get("models", [])
    except ApiError:
        return []


def predict(api_url, trip_id, model=None):
    body = {"trip_id": trip_id}
    if model:
        body["model"] = model
    return _call("POST", api_url, "/predict", timeout=300, json=body)


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

# ---------- Page choice ----------

page = st.sidebar.radio("Page", ["Reconstruct a trip", "Model comparison"])


def show_comparison(api_url):
    """Model comparison page: the API's GET /comparison tables (all from results/ CSV files)."""
    import pandas as pd

    st.header("Model comparison")
    try:
        comp = _call("GET", api_url, "/comparison", timeout=20)
    except ApiError as e:
        st.error(str(e))
        return
    st.caption(comp.get("note", ""))

    def table(key, title, best_col=None, lower_is_better=False, caption=None):
        rows = comp.get(key)
        if not rows:
            return
        df = pd.DataFrame(rows)
        st.subheader(title)
        if caption:
            st.caption(caption)
        sty = df.style.format(precision=4)
        if best_col and best_col in df:
            best = df[best_col].idxmin() if lower_is_better else df[best_col].idxmax()
            sty = sty.apply(lambda r: ["background-color: rgba(46, 160, 67, 0.25)"
                                       if r.name == best else "" for _ in r], axis=1)
        st.dataframe(sty, use_container_width=True, hide_index=True)

    table("classifiers", "Classifiers: cross-validation on the 80 train trips", "CV Top-1",
          caption="GroupKFold(5) by trip. Highlighted = best CV Top-1. Route continuity is for "
                  "the classifier alone (no HMM).")
    table("tuning", "Untuned vs tuned (Bayesian optimisation)", "CV Top-1")
    table("decoder", "Classifier only vs classifier + HMM (train, out-of-fold)", "Top-1",
          caption="Out-of-fold probabilities: every trip scored by a model that never saw it.")
    table("student_test", "Student-test (one final run)", "Top-1")


if page == "Model comparison":
    show_comparison(api_url)
    st.stop()

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
model_names = get_models(api_url)
model = None
if model_names:
    model = st.selectbox("Model (first = our final model)", model_names,
                         help="Every model runs the same pipeline: candidates, features, this "
                              "classifier, then HMM/Viterbi.")

if st.button("Reconstruct route", type="primary"):
    try:
        with st.spinner("Running the pipeline on Laptop 1..."):
            st.session_state["result"] = predict(api_url, trip_id, model)
    except ApiError as e:
        st.session_state.pop("result", None)
        st.error(str(e))

# ---------- Result (kept in session_state so map clicks don't erase it) ----------

result = st.session_state.get("result")
if result and result["trip_id"] == trip_id:
    m = result.get("metrics", {})
    timing = result.get("timing_ms", {})
    total = timing.get("total")

    st.subheader(f"Trip {result.get('trip_id', trip_id)}: {result.get('n_points', 0):,} GPS points")
    if result.get("model"):
        st.caption(f"Model used: {result['model']}")
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

    fmap = make_result_map(result)
    # center/zoom passed explicitly: st_folium does not always apply the map's fit_bounds
    st_folium(fmap, center=fmap.location, zoom=fmap.options.get("zoom"), height=600,
              use_container_width=True, returned_objects=[])

    if timing:
        st.markdown("**Time per step on Laptop 1**")
        st.table([{"step": k, "time (ms)": f"{v:,.1f}"} for k, v in timing.items()])
