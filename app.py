import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Distribution Fitter", layout="wide")
st.title("Histogram Distribution Fitter")
st.write("Upload or enter data, choose a distribution, fit it, visualize results, or manually tune parameters.")
st.sidebar.header("Data Input")
data_source = st.sidebar.radio("Select Data Input Method:",
                               ["Enter manually", "Upload CSV"])

if data_source == "Enter manually":
    user_text = st.sidebar.text_area("Enter numbers separated by commas or spaces:")
    try:
        data = np.array([float(x) for x in user_text.replace(",", " ").split()])
    except:
        data = np.array([])
else:
    file = st.sidebar.file_uploader("Upload CSV with one column of data")
    if file:
        df = pd.read_csv(file)
        data = df.iloc[:, 0].dropna().values
    else:
        data = np.array([])

if len(data) == 0:
    st.warning("Please enter or upload numerical data.")
    st.stop()

distribution_dict = {
    "Normal": stats.norm,
    "Gamma": stats.gamma,
    "Weibull": stats.weibull_min,
    "Exponential": stats.expon,
    "Lognormal": stats.lognorm,
    "Chi-square": stats.chi2,
    "Beta": stats.beta,
    "t-distribution": stats.t,
    "Pareto": stats.pareto,
    "Uniform": stats.uniform,
}

dist_name = st.sidebar.selectbox("Choose a distribution:", list(distribution_dict.keys()))
dist = distribution_dict[dist_name]

st.sidebar.header("Manual Fit Controls")
manual_mode = st.sidebar.checkbox("Enable manual parameter adjustment")
params = dist.fit(data)

if manual_mode:
    st.sidebar.write("Adjust parameters:")
    manual_params = []
    for i, p in enumerate(params):
        manual_params.append(
            st.sidebar.slider(f"Param {i} ({p:.3f})", p * 0.2, p * 5 if p != 0 else 1, p)
        )
    final_params = manual_params
else:
    final_params = params

x = np.linspace(min(data), max(data), 500)
fitted_dist = dist(*final_params)
pdf_vals = fitted_dist.pdf(x)

hist_vals, bin_edges = np.histogram(data, bins=25, density=True)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
pdf_at_centers = fitted_dist.pdf(bin_centers)

mae = np.mean(np.abs(hist_vals - pdf_at_centers))
max_err = np.max(np.abs(hist_vals - pdf_at_centers))
tab1, tab2 = st.tabs(["Visualization", "Fit Details"])

with tab1:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=25, density=True, alpha=0.5, label="Histogram")
    ax.plot(x, pdf_vals, linewidth=2, label="Fitted PDF")
    ax.set_title(f"Fitted {dist_name} Distribution")
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.subheader("Fitted Parameters")
    st.write({f"param_{i}": val for i, val in enumerate(final_params)})

    st.subheader("Error Metrics")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.6f}")
    st.write(f"**Maximum Error:** {max_err:.6f}")

st.success("Done! Adjust parameters or try a new distribution.")
