import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

@st.cache_data
def load_data():
    # Only load required columns to prevent RAM crashes
    cols = ['Sex', 'Equipment', 'BodyweightKg', 'Best3SquatKg', 
            'Best3BenchKg', 'Best3DeadliftKg', 'TotalKg', 'Federation', 'Tested', 'Dots']
    
    # Read the compressed Parquet dataset
    df = pd.read_parquet('powerlifting_data.parquet', columns=cols)
    
    # Drop rows where total, bodyweight, or ANY of the three lifts are missing
    # This guarantees we are only looking at Full Power (SBD) competitors!
    df = df.dropna(subset=[
        'TotalKg', 'Sex', 'Equipment', 'BodyweightKg', 
        'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg'
    ])
    
    return df

df = load_data()


st.title("Powerlifting Percentile Calculator")
st.markdown("Enter your numbers and adjust the filters to see exactly where you rank in powerlifting history.")

# --- ONE MASTER FORM FOR ALL SETTINGS ---
with st.sidebar.form("main_calculator_form"):
    
    # 2. UNIT TOGGLE (Now inside the form)
    st.subheader("1. Your Stats")
    unit = st.radio("Preferred Unit", ["lbs", "kg"], horizontal=True)
    is_lbs = unit == "lbs"
    
    # 3. USER INPUTS
    user_sex = st.selectbox("Sex", ["M", "F", "Mx"]) 
    user_bw = st.number_input(f"Bodyweight (In preferred unit)", min_value=0.0, value=165.0 if is_lbs else 75.0, step=1.0)
    
    st.subheader(f"Your 1RMs (In preferred unit)")
    lift_step = 5.0 if is_lbs else 2.5 
    sq = st.number_input(f"Squat", min_value=0.0, value=315.0 if is_lbs else 140.0, step=lift_step) 
    bp = st.number_input(f"Bench", min_value=0.0, value=225.0 if is_lbs else 100.0, step=lift_step) 
    dl = st.number_input(f"Deadlift", min_value=0.0, value=405.0 if is_lbs else 180.0, step=lift_step) 

    st.markdown("---")
    
    # 4. COMPARISON FILTERS
    st.subheader("2. Comparison Filters")
    equipment = st.selectbox("Equipment", ["Raw", "Wraps", "Single-ply", "Multi-ply", "All"])
    tested = st.selectbox("Drug Tested?", ["All", "Tested Only"])
    
    top_feds = df['Federation'].value_counts().index.tolist()
    federation = st.selectbox("Federation", ["All"] + top_feds)

    st.markdown("---")
    
    # 5. GRAPH SETTINGS
    st.subheader("3. Graph Settings")
    compare_scope = st.radio("Compare Against:", ["My Weight Class", "All Weight Classes"], horizontal=True)
    metric = st.radio("Scoring Metric:", ["Weight", "DOTS Score"], horizontal=True)
    
    # The crucial submit button
    submitted = st.form_submit_button("Calculate My Rank")

# --- BACKEND CONVERSION & STATS DISPLAY ---
user_total = sq + bp + dl 
user_bw_kg = user_bw / 2.20462262 if is_lbs else user_bw
user_total_kg = user_total / 2.20462262 if is_lbs else user_total

# DOTS Calculation Function
def calculate_dots(bw, total, sex):
    # Constants for DOTS Polynomial
    if sex == "F":
        a, b, c, d, e = -57.96288, 13.6175032, -0.1126655495, 0.0005158568, -0.0000010706
    else: # M and Mx use the Male curve
        a, b, c, d, e = -307.75076, 24.0900756, -0.1918759221, 0.0007391293, -0.000001093
    
    denominator = a + b * bw + c * (bw**2) + d * (bw**3) + e * (bw**4)
    return (total * 500) / denominator if denominator != 0 else 0

user_dots = calculate_dots(user_bw_kg, user_total_kg, user_sex)

# Display the user's generated stats cleanly below the form
st.sidebar.markdown(f"**Your Total:** {user_total} {unit}")
st.sidebar.markdown(f"**Your DOTS Score:** {user_dots:.2f}")

# Force extra space at the bottom of the sidebar
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

# 5. FILTERING LOGIC
mask = (df['Sex'] == user_sex)

if equipment != "All":
    mask &= (df['Equipment'] == equipment)

if tested == "Tested Only":
    mask &= (df['Tested'] == 'Yes')

if federation != "All":
    mask &= (df['Federation'] == federation)

base_filtered_df = df[mask]

# --- NEW WEIGHT CLASS LOGIC ---
# Define the upper limits of standard weight classes (in kg)
weight_classes = {
    "M": [52.0, 56.0, 60.0, 67.5, 75.0, 82.5, 90.0, 100.0, 110.0, 125.0, 140.0, float('inf')],
    "F": [44.0, 48.0, 52.0, 56.0, 60.0, 67.5, 75.0, 82.5, 90.0, 100.0, float('inf')],
    "Mx": [52.0, 56.0, 60.0, 67.5, 75.0, 82.5, 90.0, 100.0, 110.0, 125.0, 140.0, float('inf')] 
}

def get_weight_class_bounds(weight_kg, sex):
    limits = weight_classes.get(sex, weight_classes["M"])
    lower = 0.0
    for upper in limits:
        if weight_kg <= upper:
            return lower, upper
        lower = upper
    return lower, float('inf')

# Calculate exact bounds for the user
lower_kg, upper_kg = get_weight_class_bounds(user_bw_kg, user_sex)

# Filter dataframe strictly to this weight class (> lower bound, <= upper bound)
mask &= (df['BodyweightKg'] > lower_kg) & (df['BodyweightKg'] <= upper_kg)

filtered_df = df[mask]

# --- SET THE ACTIVE DATASET BASED ON TOGGLE ---
active_df = base_filtered_df.copy() if compare_scope == "All Weight Classes" else filtered_df.copy()

# 6. CALCULATE & DISPLAY PERCENTILE
st.divider()

if len(active_df) < 10:
    st.warning("Not enough data points with these exact filters. Try broadening your criteria.")
else:
    # --- DYNAMIC PERCENTILE CALCULATION ---
    if metric == "DOTS Score":
        percentile = (active_df['Dots'] < user_dots).mean() * 100
        metric_label = "DOTS points"
    else:
        percentile = (active_df['TotalKg'] < user_total_kg).mean() * 100
        metric_label = unit

    st.subheader("Your Ranking")
    st.metric(
        label=f"Compared against {len(active_df):,} lifters", 
        value=f"Top {100 - percentile:.1f}%", 
        delta=f"Higher score than {percentile:.1f}% of the field"
    )

    st.markdown(f"**Your Total:** {user_total:.1f} {unit} | **Your DOTS Score:** {user_dots:.2f} points")
    
    if compare_scope == "All Weight Classes":
        st.markdown(f"""
        *You are currently being compared to **{user_sex}** lifters across **ALL weight classes**.*
        """)
    else:
        # Convert bounds back to the user's preferred unit for the UI
        lower_display = lower_kg * 2.20462 if is_lbs else lower_kg
        upper_display = upper_kg * 2.20462 if is_lbs else upper_kg
        class_name = f"{upper_kg}kg" if upper_kg != float('inf') else f"{lower_kg}kg+"
        
        st.markdown(f"""
        *You are currently being compared to **{user_sex}** lifters in the **{class_name}** weight class 
        (weighing between **{lower_display:.1f} {unit}** and **{upper_display:.1f} {unit}**).*
        """)

# 7. VISUALIZATIONS (The Bell Curves)
    st.divider()
    st.subheader("Interactive Distributions")
    st.markdown("Hover over the curves to see percentile stats at any weight. The dashed line is you.")

    if is_lbs:
        plot_df = filtered_df[['TotalKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']] * 2.20462
    else:
        plot_df = filtered_df[['TotalKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']]
        
    def draw_interactive_curve(clean_data, user_val, lift_name, color, unit_label=unit):
        clean_data = clean_data.dropna()
        if len(clean_data) < 2:
            return None
            
        # Determine decimal formatting (2 for DOTS, 1 for Weight)
        decimals = ".2f" if unit_label == "DOTS" else ".1f"
            
        # Calculate the smooth bell curve (KDE)
        kde = gaussian_kde(clean_data)
        x_vals = np.linspace(clean_data.min(), clean_data.max(), 200)
        y_vals = kde(x_vals)
        
        # Calculate exact percentiles for the hover menu
        sorted_data = np.sort(clean_data)
        percentiles = np.searchsorted(sorted_data, x_vals) / len(sorted_data) * 100
        
        fig = go.Figure()

        # Add the interactive bell curve
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            line=dict(color=color, width=3),
            fill='tozeroy',
            opacity=0.5,
            name='Distribution',
            # Using dynamic formatting for the hover text
            hovertemplate=f"<b>Score/Weight:</b> %{{x:{decimals}}} {unit_label}<br>" +
                          "<b>Ranking:</b> Top %{customdata:.1f}%<extra></extra>",
            customdata=100 - percentiles 
        ))

        # Add a vertical dashed line for the user
        user_percentile = (clean_data < user_val).mean() * 100
        fig.add_vline(x=user_val, line_width=3, line_dash="dash", line_color="red")
        
        # Add a text annotation with exact pixel offset
        fig.add_annotation(
            x=user_val,
            y=max(y_vals),
            text=f"You: {user_val:{decimals}} {unit_label}<br>(Top {100-user_percentile:.1f}%)",
            showarrow=False,
            xanchor="left",   # Anchors the left edge of the text block to your line
            yanchor="bottom",    # Anchors the top edge of the text to the graph's peak
            xshift=15,        # Shoves the text exactly 15 pixels to the right
            yshift=-100,      # Shoves the text exactly 100 pixels down
            font=dict(color='white', size=12, family="Arial Black"),
            align="left"
        )

        fig.update_layout(
            title=f"{lift_name} Distribution",
            xaxis_title=f"Weight/Score ({unit_label})",
            yaxis_title="Density",
            yaxis=dict(showticklabels=False), 
            hovermode="x unified", 
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='lightgrey')
        )
        return fig

    # Calculate DOTS for individual lifts if requested
    if metric == "DOTS Score":
        active_unit = "DOTS"
        
        # Calculate dataset's DOTS multiplier (DOTS / TotalKg)
        df_multiplier = active_df['Dots'] / active_df['TotalKg']
        active_df['SquatDots'] = active_df['Best3SquatKg'] * df_multiplier
        active_df['BenchDots'] = active_df['Best3BenchKg'] * df_multiplier
        active_df['DeadliftDots'] = active_df['Best3DeadliftKg'] * df_multiplier
        
        # Calculate user's individual DOTS
        user_sq_kg = sq / 2.20462 if is_lbs else sq
        user_bp_kg = bp / 2.20462 if is_lbs else bp
        user_dl_kg = dl / 2.20462 if is_lbs else dl
        
        user_multiplier = user_dots / user_total_kg if user_total_kg > 0 else 0
        user_sq_val = user_sq_kg * user_multiplier
        user_bp_val = user_bp_kg * user_multiplier
        user_dl_val = user_dl_kg * user_multiplier
        user_total_val = user_dots
        
        # Assign plot data mapped to DOTS
        plot_data = active_df[['Dots', 'SquatDots', 'BenchDots', 'DeadliftDots']]
        cols_to_plot = ['Dots', 'SquatDots', 'BenchDots', 'DeadliftDots']
        
    else:
        active_unit = unit
        user_sq_val, user_bp_val, user_dl_val, user_total_val = sq, bp, dl, user_total
        
        # Assign plot data mapped to Raw Weight
        if is_lbs:
            plot_data = active_df[['TotalKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']] * 2.20462
        else:
            plot_data = active_df[['TotalKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']]
            
        cols_to_plot = ['TotalKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']

    # Draw the 4 standard tabs (removed the 5th DOTS tab)
    tab1, tab2, tab3, tab4 = st.tabs(["Total", "Squat", "Bench", "Deadlift"])
    
    with tab1:
        st.plotly_chart(draw_interactive_curve(plot_data[cols_to_plot[0]], user_total_val, "Overall Total", "#9b59b6", active_unit), use_container_width=True)
    with tab2:
        st.plotly_chart(draw_interactive_curve(plot_data[cols_to_plot[1]], user_sq_val, "Squat", "#e74c3c", active_unit), use_container_width=True)
    with tab3:
        st.plotly_chart(draw_interactive_curve(plot_data[cols_to_plot[2]], user_bp_val, "Bench Press", "#3498db", active_unit), use_container_width=True)
    with tab4:
        st.plotly_chart(draw_interactive_curve(plot_data[cols_to_plot[3]], user_dl_val, "Deadlift", "#2ecc71", active_unit), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("This page uses data from the OpenPowerlifting project, https://www.openpowerlifting.org. You may download a copy of the data at https://data.openpowerlifting.org.")