import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import firebase_admin
from firebase_admin import credentials, firestore

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Apple Upgrade Prediction Dashboard",
    page_icon="",
    layout="wide"
)

# ---------------- FIRESTORE SETUP ----------------
@st.cache_resource
def get_db():
    """Initialize Firebase Admin only once (Streamlit reruns the script frequently)."""
    if not firebase_admin._apps:
        # Read service account from [firebase] section in Streamlit secrets
        firebase_creds = dict(st.secrets["firebase"])
        cred = credentials.Certificate(firebase_creds)
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = get_db()
TARGET_COLLECTION = "apple_upgrade_predictions"


# ---------------- MODEL LOGIC ----------------
def compute_forcing_term(DA, BH, TI, ENG, PU, SI, PS):
    dt = 0.01
    eta = 0.9
    alpha = 0.7
    omega = 0.5
    t = 800

    X = np.zeros(t)
    Y = np.zeros(t)
    S = np.zeros(t)
    forcing_term = np.zeros(t)

    # initial conditions
    X[0] = alpha * (1 - BH) + (1 - alpha) * DA
    Y[0] = (omega * DA + (1 - BH) * omega) * PS
    S[0] = X[0] * (1 - Y[0])
    forcing_term[0] = 0.1

    for k in range(1, t):
        # Need
        N = (DA + TI + ENG + PU + SI) / 5.0

        # Bonding
        B = (ENG + PU + SI) / 3.0

        # Hesitation factor
        H = (
            (1 - DA) + BH + (1 - TI) + (1 - ENG)
            + (1 - PU) + (1 - SI) + PS * (1 - TI)
        ) / 7.0

        # hidden layer 2
        X[k] = alpha * B + (1 - alpha) * N - (alpha * H)
        Y[k] = (omega * N + omega * B) * H
        S[k] = X[k] * (1 - Y[k])

        # output layer
        forcing_term[k] = (
            forcing_term[k - 1]
            + eta * (S[k - 1] - forcing_term[k - 1]) * dt
        )

    return float(forcing_term[-1])


def classify_forcing_term(value: float) -> str:
    value = round(value, 2)
    if value >= 0.60:
        return "Upgrade Soon"
    elif value >= 0.10:
        return "Delay Upgrade"
    else:
        return "Churn Risk"


# ---------------- DATA LOADING FROM FIRESTORE ----------------
@st.cache_data
def load_data_from_firestore():
    docs = list(db.collection(TARGET_COLLECTION).stream())
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
            "decision": d.get("decision"),
            "created_at": d.get("created_at"),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "forcing_term" in df.columns:
        df["forcing_term"] = pd.to_numeric(df["forcing_term"], errors="coerce")
        df = df.dropna(subset=["forcing_term"])
    return df


# ---------------- GLOBAL STYLING ----------------
st.markdown(
    """
    <style>
        /* Reduce top padding */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        /* Slightly tighter sidebar */
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }
        /* Make subheaders a bit more compact */
        h3 {
            margin-top: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- MAIN APP ----------------
st.title("Apple Upgrade Prediction Dashboard")
st.caption(
    "Upload a CSV with user inputs, generate forcing terms and decisions, "
    "store them in Firestore, and explore the results interactively."
)

# Load existing data (may be empty initially)
df = load_data_from_firestore()

tab_overview, tab_segments, tab_user, tab_loader = st.tabs([
    "Overview",
    "Segment Insights",
    "User Explorer",
    "Data Loader"
])

# ========== TAB 4: DATA LOADER ==========
with tab_loader:
    st.subheader("Data Loader (CSV → Firestore)")

    st.markdown(
        """
        This panel lets you take a raw dataset, run it through the forcing-term model,
        and push the results into the Firestore collection used by the dashboard.

        The CSV must include at least these columns:

        `id`, `DA`, `BH`, `TI`, `ENG`, `PU`, `SI`, `PS`
        """
    )

    uploaded_file = st.file_uploader("Select a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Unable to read CSV file: {e}")
            raw_df = None

        if raw_df is not None:
            with st.expander("Preview uploaded data", expanded=True):
                st.dataframe(raw_df.head())

            required_cols = ["id", "DA", "BH", "TI", "ENG", "PU", "SI", "PS"]
            missing = [c for c in required_cols if c not in raw_df.columns]

            if missing:
                st.error(f"The following required columns are missing: {missing}")
            else:
                if st.button("Compute and save to Firestore"):
                    with st.spinner("Computing forcing terms and writing documents..."):
                        processed_count = 0
                        for _, row in raw_df.iterrows():
                            try:
                                user_id = str(row["id"])
                                DA = float(row["DA"])
                                BH = float(row["BH"])
                                TI = float(row["TI"])
                                ENG = float(row["ENG"])
                                PU = float(row["PU"])
                                SI = float(row["SI"])
                                PS = float(row["PS"])

                                raw_value = compute_forcing_term(
                                    DA, BH, TI, ENG, PU, SI, PS
                                )
                                forcing_value = round(raw_value, 3)
                                decision = classify_forcing_term(forcing_value)

                                out_doc = {
                                    "DA": DA,
                                    "BH": BH,
                                    "TI": TI,
                                    "ENG": ENG,
                                    "PU": PU,
                                    "SI": SI,
                                    "PS": PS,
                                    "forcing_term": forcing_value,
                                    "decision": decision,
                                    "source_id": user_id,
                                    "created_at": firestore.SERVER_TIMESTAMP,
                                }

                                db.collection(TARGET_COLLECTION).document(user_id).set(out_doc)
                                processed_count += 1
                            except Exception as e:
                                st.warning(
                                    f"Row with id={row.get('id', 'N/A')} "
                                    f"was skipped due to an error: {e}"
                                )

                        # Clear cached Firestore data so new rows appear
                        load_data_from_firestore.clear()
                        st.success(f"Completed. {processed_count} rows were written to Firestore.")
                        st.info("Use the other tabs to explore the updated dataset.")

# If no data yet, stop other tabs early
if df.empty:
    with tab_overview:
        st.warning(
            "No computed records found in Firestore. "
            "Use the Data Loader tab to upload and process a CSV first."
        )
    with tab_segments:
        st.info("Segment-level summaries will be available once data exists.")
    with tab_user:
        st.info("User-level exploration will be available once data exists.")
    st.stop()

# ---------------- FILTERS (shared by tabs, only if data exists) ----------------
st.sidebar.title("Filters")

decision_options = ["Upgrade Soon", "Delay Upgrade", "Churn Risk"]
selected_decisions = st.sidebar.multiselect(
    "Decision segment",
    options=decision_options,
    default=decision_options,
)

forcing_min_val = float(df["forcing_term"].min())
forcing_max_val = float(df["forcing_term"].max())

forcing_min, forcing_max = st.sidebar.slider(
    "Forcing term range",
    forcing_min_val,
    forcing_max_val,
    (forcing_min_val, forcing_max_val),
    step=0.05,
)

filtered_df = df[
    df["decision"].isin(selected_decisions)
    & (df["forcing_term"] >= forcing_min)
    & (df["forcing_term"] <= forcing_max)
].copy()

total_users = len(filtered_df)
avg_forcing = filtered_df["forcing_term"].mean() if total_users else 0.0

upgrade_count = int((filtered_df["decision"] == "Upgrade Soon").sum())
delay_count = int((filtered_df["decision"] == "Delay Upgrade").sum())
churn_count = int((filtered_df["decision"] == "Churn Risk").sum())

upgrade_rate = (upgrade_count / total_users * 100) if total_users else 0
delay_rate = (delay_count / total_users * 100) if total_users else 0
churn_rate = (churn_count / total_users * 100) if total_users else 0

# ========== TAB 1: OVERVIEW ==========
with tab_overview:
    st.subheader("Overview")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total users (filtered)", total_users)
    with kpi2:
        st.metric("Average forcing term", f"{avg_forcing:.3f}")
    with kpi3:
        st.metric("Upgrade Soon", f"{upgrade_rate:.1f}%")
    with kpi4:
        st.metric("Delay Upgrade", f"{delay_rate:.1f}%")

    st.write(f"Churn Risk: {churn_count} users ({churn_rate:.1f}%)")

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Forcing term by user")
        if not filtered_df.empty:
            line_df = (
                filtered_df.sort_values("forcing_term")
                .set_index("id")[["forcing_term"]]
            )
            st.line_chart(line_df)
        else:
            st.info("No data is available for the current filter combination.")

    with col_right:
        st.subheader("Decision breakdown")
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
                    startangle=90,
                )
                ax.axis("equal")
                st.pyplot(fig)
            else:
                st.info("No data is available for the current filter combination.")
        else:
            st.info("No data is available for the current filter combination.")

    st.markdown("### Forcing term distribution")
    if not filtered_df.empty:
        arr = filtered_df["forcing_term"].to_numpy()
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(arr, bins=10, edgecolor="black")
        ax_hist.set_xlabel("Forcing term")
        ax_hist.set_ylabel("Frequency")
        ax_hist.set_title("Distribution of forcing term")
        st.pyplot(fig_hist)
    else:
        st.info("No data is available for the current filter combination.")

# ========== TAB 2: SEGMENT INSIGHTS ==========
with tab_segments:
    st.subheader("Segment Insights")

    feature_cols = ["DA", "BH", "TI", "ENG", "PU", "SI", "PS", "forcing_term"]

    seg_df = filtered_df.copy()
    seg_df[feature_cols] = seg_df[feature_cols].apply(pd.to_numeric, errors="coerce")

    if seg_df.empty:
        st.info("Not enough data to compute segment statistics for the current filters.")
    else:
        seg_stats = (
            seg_df
            .groupby("decision")[feature_cols]
            .mean()
            .reindex(decision_options)
        )

        if "Upgrade Soon" in seg_stats.index:
            upgrade_means = seg_stats.loc[
                "Upgrade Soon", feature_cols[:-1]
            ]  # exclude forcing_term
            upgrade_means = upgrade_means.dropna()
            if not upgrade_means.empty:
                top_driver = upgrade_means.idxmax()
                st.info(
                    f"For users likely to upgrade soon, the strongest average driver "
                    f"among the input features is: {top_driver} "
                    f"(mean value {upgrade_means.max():.3f})."
                )

        st.markdown("### Feature comparison across segments")

        selected_feature = st.selectbox(
            "Feature to compare",
            options=feature_cols,
            index=feature_cols.index("forcing_term"),
        )

        feat_df = seg_stats[[selected_feature]].reset_index()
        feat_df.rename(columns={selected_feature: "value"}, inplace=True)

        st.bar_chart(
            data=feat_df,
            x="decision",
            y="value",
            use_container_width=True,
        )

        st.markdown("### Correlation between inputs and forcing term")
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
with tab_user:
    st.subheader("User Explorer")

    if filtered_df.empty:
        st.info("No users match the current filter settings.")
    else:
        selected_user_id = st.selectbox(
            "Select a user identifier",
            options=filtered_df["id"].tolist(),
        )

        user_row = filtered_df[filtered_df["id"] == selected_user_id].iloc[0]

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("Readiness overview")
            st.metric("Decision", user_row["decision"])
            st.metric("Forcing term", f"{user_row['forcing_term']:.3f}")
            norm_value = np.clip((user_row["forcing_term"] + 0.2) / 1.0, 0, 1)
            st.progress(float(norm_value))

        with col2:
            st.markdown("Input profile")

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
                ],
            })

            st.bar_chart(
                data=profile_df,
                x="Feature",
                y="Value",
                use_container_width=True,
            )

        st.markdown("---")
        st.caption(
            "Use the filters in the sidebar to adjust the population, then inspect "
            "individual records here."
        )
