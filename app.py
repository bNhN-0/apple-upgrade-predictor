import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import firebase_admin
from firebase_admin import credentials, firestore

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Apple Upgrade Prediction Dashboard",
    page_icon="🍏",
    layout="wide"
)

# ---------------- FIRESTORE SETUP ----------------
@st.cache_resource
def get_db():
    """Initialize Firebase Admin only once (Streamlit reruns the script a lot)."""
    if not firebase_admin._apps:
        # Convert Streamlit secrets section to a plain dict for firebase_admin
        firebase_creds = dict(st.secrets["firebase"])
        cred = credentials.Certificate(firebase_creds)
        firebase_admin.initialize_app(cred)
    return firestore.client()
db = get_db()
COLLECTION_NAME = "apple_upgrade_predictions"

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    docs = list(db.collection(COLLECTION_NAME).stream())
    rows = []
    for doc in docs:
        d = doc.to_dict()
        rows.append({
            "id": d.get("source_id", doc.id),
            "DA": d.get("DA"),
            "BH": d.get("BH"),
            "TI": d.get("TI"),
            "ENG": d.get("ENG"),
            "PU": d.get("PU"),
            "SI": d.get("SI"),
            "PS": d.get("PS"),
            "forcing_term": d.get("forcing_term"),
            "decision": d.get("decision")
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

df = load_data()

# Ensure forcing_term is numeric and valid
if "forcing_term" in df.columns:
    df["forcing_term"] = pd.to_numeric(df["forcing_term"], errors="coerce")
    df = df.dropna(subset=["forcing_term"])

# ---------------- EMPTY STATE ----------------
if df.empty:
    st.title("🍏 Apple Upgrade Prediction Dashboard")
    st.error("No documents found in Firestore or forcing_term values are invalid. Run your batch script to push data first.")
    st.stop()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.title("🔎 Filters")

decision_options = ["Upgrade Soon", "Delay Upgrade", "Churn Risk"]
selected_decisions = st.sidebar.multiselect(
    "Decision segment",
    options=decision_options,
    default=decision_options  # show all by default
)

forcing_min_val = float(df["forcing_term"].min())
forcing_max_val = float(df["forcing_term"].max())

forcing_min, forcing_max = st.sidebar.slider(
    "Forcing term range",
    forcing_min_val,
    forcing_max_val,
    (forcing_min_val, forcing_max_val),
    step=0.05
)

# Apply filters
filtered_df = df[
    df["decision"].isin(selected_decisions) &
    (df["forcing_term"] >= forcing_min) &
    (df["forcing_term"] <= forcing_max)
].copy()

# ---------------- HEADER ----------------
st.title("Apple Upgrade Prediction Dashboard")
st.caption(f"Data source: Firestore collection `{COLLECTION_NAME}`")

# ---------------- KPI CARDS ----------------
total_users = len(filtered_df)
avg_forcing = filtered_df["forcing_term"].mean() if total_users else 0.0

upgrade_count = int((filtered_df["decision"] == "Upgrade Soon").sum())
delay_count   = int((filtered_df["decision"] == "Delay Upgrade").sum())
churn_count   = int((filtered_df["decision"] == "Churn Risk").sum())

upgrade_rate = (upgrade_count / total_users * 100) if total_users else 0
delay_rate   = (delay_count   / total_users * 100) if total_users else 0
churn_rate   = (churn_count   / total_users * 100) if total_users else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Total Users (filtered)", total_users)
with kpi2:
    st.metric("Avg. Forcing Term", f"{avg_forcing:.3f}")
with kpi3:
    st.metric("Upgrade Soon (%)", f"{upgrade_rate:.1f}%")
with kpi4:
    st.metric("Delay Upgrade (%)", f"{delay_rate:.1f}%")

st.write(f"Churn Risk users: **{churn_count}** ({churn_rate:.1f}%)")

st.markdown("---")

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🧩 Segment Insights", "👤 User Explorer"])

# ========== TAB 1: OVERVIEW ==========
with tab1:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Forcing Term by User")
        if not filtered_df.empty:
            line_df = filtered_df.sort_values("forcing_term").set_index("id")[["forcing_term"]]
            st.line_chart(line_df)
        else:
            st.info("No data available for current filters.")

    with col_right:
        st.subheader("Decision Breakdown")

        if not filtered_df.empty:
            decision_counts = (
                filtered_df["decision"]
                .value_counts()
                .reindex(decision_options, fill_value=0)
            )

            fig, ax = plt.subplots()
            labels = decision_counts.index.tolist()
            sizes = decision_counts.values.tolist()

            if sizes and sum(sizes) > 0:
                ax.pie(
                    sizes,
                    labels=labels,
                    autopct="%1.0f%%",
                    startangle=90
                )
                ax.axis("equal")
                st.pyplot(fig)
            else:
                st.info("No data available for current filters.")
        else:
            st.info("No data available for current filters.")

    st.markdown("### Forcing Term Distribution")
    if not filtered_df.empty:
        arr = filtered_df["forcing_term"].to_numpy()
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(arr, bins=10, edgecolor="black")
        ax_hist.set_xlabel("Forcing term value")
        ax_hist.set_ylabel("Frequency")
        ax_hist.set_title("Distribution of forcing_term")
        st.pyplot(fig_hist)
    else:
        st.info("No data available for current filters.")

# ========== TAB 2: SEGMENT INSIGHTS ==========
with tab2:
    st.subheader("Segment Insights")

    feature_cols = ["DA", "BH", "TI", "ENG", "PU", "SI", "PS", "forcing_term"]

    seg_df = filtered_df.copy()
    seg_df[feature_cols] = seg_df[feature_cols].apply(pd.to_numeric, errors="coerce")

    if seg_df.empty:
        st.info("Not enough data to compute segment insights for current filters.")
    else:
        seg_stats = (
            seg_df
            .groupby("decision")[feature_cols]
            .mean()
            .reindex(decision_options)  # consistent order
        )

        # --- Highlight key insight ---
        if "Upgrade Soon" in seg_stats.index:
            upgrade_means = seg_stats.loc["Upgrade Soon", feature_cols[:-1]]  # exclude forcing_term
            upgrade_means = upgrade_means.dropna()
            if not upgrade_means.empty:
                top_driver = upgrade_means.idxmax()
                st.info(
                    f"**Top driver for 'Upgrade Soon' users:** `{top_driver}` "
                    f"(avg = {upgrade_means.max():.3f})"
                )

        st.markdown("### Compare Features Across Segments")

        selected_feature = st.selectbox(
            "Select a feature to compare:",
            options=feature_cols,
            index=feature_cols.index("forcing_term")
        )

        feat_df = seg_stats[[selected_feature]].reset_index()
        feat_df.rename(columns={selected_feature: "value"}, inplace=True)

        st.bar_chart(
            data=feat_df,
            x="decision",
            y="value",
            use_container_width=True
        )

        st.markdown("### Inputs vs Forcing Term (Correlation Heatmap)")
        corr = seg_df[feature_cols].corr()

        fig_corr, ax_corr = plt.subplots()
        cax = ax_corr.imshow(corr, interpolation="nearest")
        ax_corr.set_xticks(range(len(feature_cols)))
        ax_corr.set_yticks(range(len(feature_cols)))
        ax_corr.set_xticklabels(feature_cols, rotation=45, ha="right")
        ax_corr.set_yticklabels(feature_cols)
        fig_corr.colorbar(cax)
        st.pyplot(fig_corr)

# ========== TAB 3: USER EXPLORER ==========
with tab3:
    st.subheader("User Explorer")

    if filtered_df.empty:
        st.info("No users match the current filters.")
    else:
        selected_user_id = st.selectbox(
            "Select a user ID:",
            options=filtered_df["id"].tolist()
        )

        user_row = filtered_df[filtered_df["id"] == selected_user_id].iloc[0]

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Upgrade Readiness")

            st.metric("Decision", user_row["decision"])
            st.metric("Forcing Term", f"{user_row['forcing_term']:.3f}")

            # Simple normalized readiness bar
            norm_value = np.clip((user_row["forcing_term"] + 0.2) / 1.0, 0, 1)
            st.progress(float(norm_value))

        with col2:
            st.markdown("#### Input Profile (Feature Strengths)")

            profile_df = pd.DataFrame({
                "Feature": ["DA", "BH", "TI", "ENG", "PU", "SI", "PS"],
                "Value": [
                    user_row["DA"],
                    user_row["BH"],
                    user_row["TI"],
                    user_row["ENG"],
                    user_row["PU"],
                    user_row["SI"],
                    user_row["PS"],
                ]
            })

            st.bar_chart(
                data=profile_df,
                x="Feature",
                y="Value",
                use_container_width=True
            )

        st.markdown("---")
        st.caption(
            "Tip: Use the filters in the sidebar to narrow down to Upgrade, Delay, or Churn segments, "
            "then inspect individual users here."
        )
