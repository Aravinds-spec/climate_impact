import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter

# Define the Biome Thresholds globally (NCI thresholds for rejection)
BIOME_THRESHOLDS = {
    'Tropical & Subtropical Dry Broadleaf Forests': 18.5,
    'Tropical & Subtropical Moist Broadleaf Forests': 8.5,
    'Deserts & Xeric Shrublands': 39.5
}

# --- Data Preparation Functions for NCI Data (File 1) ---

@st.cache_data
def load_nci_data(uploaded_file):
    """Loads and preprocesses the NCI DataFrame."""
    if uploaded_file is None:
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        if hasattr(uploaded_file, 'getvalue'):
            try:
                # Attempt to read with a custom separator if standard CSV fails
                df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')), sep='|')
            except Exception:
                st.error("NCI File: Failed to read as CSV or custom-separated format.")
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    # --- CRITICAL: Column Assignment based on Sample Data ---
    numerical_cols = df.select_dtypes(include=np.number).columns
    
    # We need to ensure 'BIOME_NAME' column exists for NCI analysis
    if 'BIOME_NAME' not in df.columns:
        st.error("NCI File Error: Missing required column 'BIOME_NAME'.")
        return pd.DataFrame()

    # Assuming Area_Ha and NCI_Score are the last two numerical columns
    if len(numerical_cols) >= 2:
        # Assuming the second to last is Area_Ha, and the last is NCI_Score
        df['Area_Ha'] = df[numerical_cols[-2]]
        df['NCI_Score'] = df[numerical_cols[-1]]
    else:
        st.error("NCI File Error: Could not identify the two required numerical columns (Area_Ha and NCI_Score). Please check your file.")
        return pd.DataFrame()

    # Preprocessing
    df['NCI_Score_Rounded'] = df['NCI_Score'].round(2)
    
    # 1. Map the NCI Threshold to each row based on its BIOME_NAME
    df['NCI_Threshold'] = df['BIOME_NAME'].apply(
        lambda x: BIOME_THRESHOLDS.get(x, 0)
    )

    # 2. Determine rejection status
    df['is_rejected'] = df['NCI_Score'] < df['NCI_Threshold']
    
    # 3. Assign the final Status Label
    df['Status_Label'] = df['is_rejected'].apply(
        lambda x: 'REJECTED (NCI < Threshold)' if x else 'ACCEPTED (NCI >= Threshold)'
    )

    return df.copy()

# --- Summary and Plotting Functions for NCI Data ---

def get_overall_rejection_summary(data_df):
    """Calculates the overall summary focused on NCI Rejection, using area for the rejection rate."""
    rejected_df = data_df[data_df['is_rejected'] == True]
    accepted_df = data_df[data_df['is_rejected'] == False]
    
    total_records = len(data_df)
    total_rejected = len(rejected_df)
    
    # --- Robust Area Calculation ---
    total_area_rejected = round(rejected_df['Area_Ha'].sum(), 2) if not rejected_df.empty else 0.0
    total_area_accepted = round(accepted_df['Area_Ha'].sum(), 2) if not accepted_df.empty else 0.0
    total_farm_area = round(data_df['Area_Ha'].sum(), 2)
    
    # --- Area-Based Rejection Rate Calculation ---
    area_rejection_rate = 0
    if total_farm_area > 0:
        area_rejection_rate = round((total_area_rejected / total_farm_area) * 100, 2)

    overall_summary = {
        'Total Records': total_records,
        'Records Rejected': total_rejected,
        'Records Accepted': total_records - total_rejected,
        'Total Area (Ha) Under Analysis': total_farm_area,
        'Area Rejection Rate (%)': area_rejection_rate,
        'Total Area Rejected (Ha)': total_area_rejected,
        'Total Area Accepted (Ha)': total_area_accepted,
        'Avg NCI Score (Rejected)': round(rejected_df['NCI_Score'].mean(), 2) if not rejected_df.empty else 0.0,
    }
    return overall_summary

def get_grouped_status_summary(data_df, group_col):
    """Calculates acceptance/rejection count and area summary by a grouping column (Biome/State)."""
    
    REJECTED_LABEL = 'REJECTED (NCI < Threshold)'
    ACCEPTED_LABEL = 'ACCEPTED (NCI >= Threshold)'
    
    # 1. Group and Unstack (Creates MultiIndex Columns)
    summary = data_df.groupby([group_col, 'Status_Label']).agg(
        Count=('Name', 'count'), 
        Total_Area_Ha=('Area_Ha', 'sum')
    ).unstack(fill_value=0)
    
    # --- 2. FINAL FIX: Create a simple, single-level Index from the MultiIndex components ---
    
    final_names = ['Rejected_Count', 'Accepted_Count', 'Rejected_Area_Ha', 'Accepted_Area_Ha']
    source_cols = [
        ('Count', REJECTED_LABEL), ('Count', ACCEPTED_LABEL),
        ('Total_Area_Ha', REJECTED_LABEL), ('Total_Area_Ha', ACCEPTED_LABEL)
    ]
    
    # Ensure all required columns exist before filtering/renaming
    for col in source_cols:
        if col not in summary.columns:
            # Add missing column with zeros
            summary[col] = 0
    
    summary = summary[source_cols]
    summary.columns = final_names
    
    # --- 3. Final Calculations ---
    summary['Total_Records'] = summary['Rejected_Count'] + summary['Accepted_Count']
    summary['Total_Area_Ha'] = summary['Rejected_Area_Ha'] + summary['Accepted_Area_Ha']
    
    # Area-Based Rejection Rate
    summary['Area_Rejection_Rate_%'] = (summary['Rejected_Area_Ha'] / summary['Total_Area_Ha'] * 100).round(1).fillna(0)

    # Return the required columns
    return summary[['Total_Records', 'Total_Area_Ha', 'Rejected_Area_Ha', 
                    'Accepted_Area_Ha', 'Area_Rejection_Rate_%']].sort_values(by='Rejected_Area_Ha', ascending=False)

# --- Data Preparation Functions for NDVI Data (File 2) ---

@st.cache_data
def load_ndvi_data(uploaded_file):
    """Loads and preprocesses the NDVI DataFrame."""
    if uploaded_file is None:
        return pd.DataFrame()
    
    try:
        df_d = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"NDVI File: Failed to read CSV. Error: {e}")
        return pd.DataFrame()

    col_lis_ = [col for col in df_d.columns if '20' in col]
    col_mapping = {}

    for old_col_name in col_lis_:
        try:
            dt_object = datetime.strptime(old_col_name, '%b_%Y')
            new_col_name = dt_object.strftime('%Y-%m-%d') # Use YYYY-MM-DD format
            col_mapping[old_col_name] = new_col_name
        except ValueError:
            pass # Skip non-date columns

    if col_mapping:
        df_d.rename(columns=col_mapping, inplace=True)
    
    # Get the new list of NDVI columns in correct format
    col_lis = [col for col in df_d.columns if '20' in col]

    # Check for 'kyari_id' as the key column
    if 'kyari_id' not in df_d.columns:
        st.error("NDVI File Error: Missing required primary key column 'kyari_id'.")
        return pd.DataFrame()

    # Apply Savitzky-Golay filter and scaling (your existing logic)
    if col_lis:
        df_ = df_d[col_lis].apply(
            lambda row: savgol_filter(row.fillna(0), window_length=7, polyorder=2),
            axis=1,
            result_type='broadcast'
        )
        df_d[col_lis] = df_
        df_d[col_lis] = df_d[col_lis] / 1000 # Scale values

    return df_d.copy(), col_lis

# --- Crop Cycle Analysis Functions ---
# (Using the original logic from the user's second script)

def analyze_crop_cycles(row, ndvi_columns, ndvi_start_threshold=0.2, ndvi_end_threshold=None, min_months=2, max_months=12):
    """Analyzes a single row of NDVI time-series data to find valid crop cycles."""
    if ndvi_end_threshold is None:
        ndvi_end_threshold = ndvi_start_threshold
            
    dates = [datetime.strptime(col, '%Y-%m-%d') for col in ndvi_columns]  
    ndvi_values = pd.to_numeric(row[ndvi_columns], errors='coerce').fillna(0).values

    cycles = []
    start_index = None
    in_growth_cycle = False 

    for i, ndvi in enumerate(ndvi_values):
        if ndvi > ndvi_start_threshold and not in_growth_cycle:
            start_index = i
            in_growth_cycle = True
            
        elif in_growth_cycle and ndvi <= ndvi_end_threshold:
            end_index = i - 1 
            
            if start_index <= end_index: 
                start_date = dates[start_index]
                end_date = dates[end_index]
                duration_months = (end_date - start_date).days / 30.0

                if duration_months >= min_months:
                    is_exception = duration_months > max_months
                    max_ndvi_in_cycle = None if is_exception else round(ndvi_values[start_index:end_index + 1].max(), 2)
                    
                    cycles.append({
                        'start_index': start_index,
                        'end_index': end_index,
                        'start_date': start_date.strftime('%Y-%m-%d'),
                        'end_date': end_date.strftime('%Y-%m-%d'),
                        'duration_months': round(duration_months, 1),
                        'max_ndvi': max_ndvi_in_cycle,
                        'is_exception': is_exception
                    })
            
            start_index = None
            in_growth_cycle = False

    # Check for ongoing cycle
    if in_growth_cycle and start_index is not None:
        end_index = len(dates) - 1 
        end_date = dates[-1]
        start_date = dates[start_index]
        duration_months = (end_date - start_date).days / 30.0
        
        if duration_months >= min_months:
            is_exception = duration_months > max_months
            max_ndvi_in_cycle = None if is_exception else round(ndvi_values[start_index:].max(), 2)
            cycles.append({
                'start_index': start_index,
                'end_index': end_index,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d') + ' (Ongoing)',
                'duration_months': round(duration_months, 1),
                'max_ndvi': max_ndvi_in_cycle,
                'is_exception': is_exception
            })
            
    return cycles

@st.cache_data
def process_ndvi_dataframe(df, ndvi_cols, ndvi_start_threshold=0.2, ndvi_end_threshold=None):
    """Applies the crop cycle analysis to each row of the DataFrame and aggregates results."""
    
    # Ensure a copy to avoid SettingWithCopyWarning if not already a copy
    df_copy = df.copy() 
    results = []
    ndvi_over_04_counts = []

    for _, row in df_copy.iterrows():
        cycles_found = analyze_crop_cycles(row, ndvi_cols, ndvi_start_threshold=ndvi_start_threshold, ndvi_end_threshold=ndvi_end_threshold) 
        results.append(cycles_found)
        
        ndvi_row_values = pd.to_numeric(row[ndvi_cols], errors='coerce').fillna(0).values
        count_over_04 = np.sum(ndvi_row_values > 0.4)
        ndvi_over_04_counts.append(count_over_04)
        
    df_copy['Num_Cycles'] = [len([c for c in r if not c['is_exception']]) for r in results]
    df_copy['Max_NDVI_Cycles'] = [[c['max_ndvi'] for c in r if not c['is_exception']] for r in results]
    df_copy['Exception_Cycles_Count'] = [sum(1 for c in r if c['is_exception']) for r in results]
    df_copy['Total_Cycles_Summary'] = df_copy.apply(
        lambda row: f"{row['Num_Cycles']} Valid, {row['Exception_Cycles_Count']} Exception",
        axis=1
    )
    df_copy['NDVI_Obs_Over_0.4'] = ndvi_over_04_counts
    df_copy['Detailed_Cycles'] = results
    
    return df_copy

def plot_crop_cycle_analysis(row, ndvi_columns, ndvi_start_threshold=0.2, ndvi_end_threshold=None):
    """Generates the Matplotlib plot for a single field's NDVI time series."""
    if ndvi_end_threshold is None:
        ndvi_end_threshold = ndvi_start_threshold
        
    field_name = row.get('Name', row.get('kyari_id', 'Selected Field'))
    
    dates = [datetime.strptime(col, '%Y-%m-%d') for col in ndvi_columns]
    ndvi_values = pd.to_numeric(row[ndvi_columns], errors='coerce').fillna(0).values
    cycles = row['Detailed_Cycles']

    fig, ax = plt.subplots(figsize=(12, 6)) # Use figure and axes objects for Streamlit

    # 1. Plot raw NDVI values
    ax.plot(dates, ndvi_values, marker='o', linestyle='-', color='#1f77b4', label='NDVI Value')
    
    # 2. Plot threshold lines
    ax.axhline(y=ndvi_start_threshold, color='r', linestyle='--', label=f'Start Threshold ({ndvi_start_threshold})')
    if ndvi_start_threshold != ndvi_end_threshold:
        ax.axhline(y=ndvi_end_threshold, color='purple', linestyle=':', label=f'End Threshold ({ndvi_end_threshold})')

    # 3. Highlight detected cycles
    cycle_index = 1
    for cycle in cycles:
        start_idx = cycle['start_index']
        end_idx = cycle['end_index']
        
        color = 'g' if not cycle['is_exception'] else 'orange'
        
        cycle_dates = dates[start_idx : end_idx + 1]
        cycle_ndvi = ndvi_values[start_idx : end_idx + 1]
        
        ax.plot(cycle_dates, cycle_ndvi, marker='o', linestyle='-', 
                        linewidth=4, color=color, alpha=0.7) 
        
        # 4. Add annotation for max NDVI
        if not cycle['is_exception']:
            max_val = cycle['max_ndvi']
            # Find the index of the max value within the current segment
            max_idx_in_segment = np.argmax(cycle_ndvi)
            max_date = cycle_dates[max_idx_in_segment]
            
            ax.annotate(f'{max_val}', (max_date, max_val),
                        textcoords="offset points", xytext=(0, 10), ha='center',
                        fontsize=9, color='black', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))

        cycle_index += 1

    # 5. Final plot formatting
    ax.set_title(f'Crop Cycle Analysis for {field_name}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Difference Vegetation Index (NDVI)')
    ax.set_ylim(0, 0.8)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Re-plot a dummy line for legend clarity 
    ax.plot([], [], linewidth=4, color='g', alpha=0.7, label='Valid Cycle')
    ax.plot([], [], linewidth=4, color='orange', alpha=0.7, label='Exception Cycle')
    
    ax.legend()
    fig.autofmt_xdate(rotation=45) # Use figure's method for date rotation
    plt.tight_layout()
    
    return fig # Return the figure object

# --- STREAMLIT APP LAYOUT (Unified) ---

st.set_page_config(layout="wide", page_title="Integrated Farm Analysis Dashboard", initial_sidebar_state="expanded")

st.title("🌱 Integrated Farm Analysis Dashboard")
st.markdown("### NCI Rejection Status & Crop Cycle Analysis")

# --- File Uploader Section ---
st.sidebar.header("Data Uploads")
nci_uploaded_file = st.sidebar.file_uploader("1. Upload NCI/Area Data CSV (Primary)", type=["csv"])
ndvi_uploaded_file = st.sidebar.file_uploader("2. Upload NDVI Time Series Data (Secondary)", type=["csv"])

# Define thresholds for NDVI analysis
NDVI_START_THRESHOLD = st.sidebar.slider("Start Threshold", 0.0, 0.5, 0.3, 0.01)
NDVI_END_THRESHOLD = st.sidebar.slider("End Threshold", 0.0, 0.5, 0.4, 0.01)

# --- Dynamic Threshold for Summary ---
NDVI_OBS_THRESHOLD = st.sidebar.slider(
    "Observation Threshold (Filter for Section 3)",
    min_value=1, max_value=200, value=10, step=1
)

if nci_uploaded_file is not None:
    nci_df = load_nci_data(nci_uploaded_file)
    nci_df['Primary_Key'] = nci_df['Name'] # Use 'Name' as the common key (based on request)
    is_nci_data_ready = not nci_df.empty
else:
    nci_df = pd.DataFrame()
    is_nci_data_ready = False

if ndvi_uploaded_file is not None:
    ndvi_raw_df, ndvi_cols = load_ndvi_data(ndvi_uploaded_file)
    
    # Process the NDVI data with the selected thresholds
    processed_ndvi_df = process_ndvi_dataframe(ndvi_raw_df, ndvi_cols, 
                                               ndvi_start_threshold=NDVI_START_THRESHOLD, 
                                               ndvi_end_threshold=NDVI_END_THRESHOLD)
    
    # Set 'kyari_id' as the common key for the NDVI data
    processed_ndvi_df['Primary_Key'] = processed_ndvi_df['kyari_id'] 
    
    is_ndvi_data_ready = not processed_ndvi_df.empty and ndvi_cols
else:
    processed_ndvi_df = pd.DataFrame()
    ndvi_cols = []
    is_ndvi_data_ready = False


# --- Display NCI Analysis ---
if is_nci_data_ready:
    
    # 1. OVERALL NCI REJECTION SUMMARY
    overall_summary = get_overall_rejection_summary(nci_df)
    
    st.header("1. Overall NCI Status Summary")
    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
    
    col_m1.metric("Total Records Analyzed", overall_summary['Total Records'])
    col_m2.metric("Total Area (Ha) Under Analysis", overall_summary['Total Area (Ha) Under Analysis'])
    col_m3.metric("Total Records Rejected", overall_summary['Records Rejected'])
    col_m4.metric("Area Rejection Rate (%)", f"{overall_summary['Area Rejection Rate (%)']}%") 
    col_m5.metric("Total Area Rejected (Ha)", overall_summary['Total Area Rejected (Ha)'])
    col_m6.metric("Total Area Accepted (Ha)", overall_summary['Total Area Accepted (Ha)'])
    

    # 2. BIOME-BASED REJECTION ANALYSIS
    st.header("2. Biome-Name Based Rejection Analysis")
    biome_status_df = get_grouped_status_summary(nci_df, 'BIOME_NAME')
    
    col_b1, col_b2 = st.columns([1, 1])
    
    with col_b1:
        st.subheader("Area Summary by Biome")
        st.dataframe(biome_status_df, use_container_width=True)
        
    with col_b2:
        st.subheader("Area Status Distribution by Biome")
        fig_biome_area = px.bar(
            nci_df,
            x='BIOME_NAME',
            y='Area_Ha',
            color='Status_Label',
            title='Area (Ha) Rejected vs. Accepted by Biome',
            color_discrete_map={'REJECTED (NCI < Threshold)': 'orange', 'ACCEPTED (NCI >= Threshold)': 'blue'}
        )
        fig_biome_area.update_layout(xaxis={'categoryorder': 'total descending'}, yaxis_title="Total Area (Ha)")
        st.plotly_chart(fig_biome_area, use_container_width=True)
    st.markdown("---")
    
    # 3. DATA LOADED SUCCESSFULLY
    st.header("NCI Data")
    st.success(f"Successfully loaded and processed {len(nci_df)} records from the uploaded NCI file.")
    st.dataframe(nci_df[['BIOME_NAME', 'District', 'Name', 'State', 'Area_Ha', 'NCI_Score', 'NCI_Threshold', 'Status_Label']], use_container_width=True)
    st.markdown("---")


# --- SECTION 3: Summary of NCI and Farm history (Aggregation FIX applied here) ---

if is_nci_data_ready and is_ndvi_data_ready:
    
    st.header("3. Summary of NCI and Farm history")
    
    # --- CRITICAL FIX: Aggregate NDVI data to one row per kyari_id using MEDIAN ---
    # 1. Aggregate the processed NDVI data to get one median value per farm
    # Assuming 'Primary_Key' is the correct kyari identifier.
    ndvi_summary_agg = processed_ndvi_df.pivot_table(
        index='Primary_Key',
        values='NDVI_Obs_Over_0.4',
        aggfunc='median'
    ).reset_index()
    
    ndvi_summary_agg.rename(columns={'NDVI_Obs_Over_0.4': 'Median_NDVI_Obs_Over_0.4'}, inplace=True)
    # -----------------------------------------------------------------------------
    
    # Merge NCI data with the aggregated NDVI features for summary calculation
    summary_df = pd.merge(
        nci_df, 
        ndvi_summary_agg, # Use the aggregated data frame
        left_on='Primary_Key', 
        right_on='Primary_Key',
        how='inner' # Use inner merge to only count farms present in both
    )
    
    # 1. Filter for the farms rejected by NCI
    rejected_nci_df = summary_df[summary_df['is_rejected'] == True]
    
    # 2. Filter for farms where the MEDIAN NDVI_Obs_Over_0.4 is less than the dynamic threshold
    rejected_both_df = rejected_nci_df[rejected_nci_df['Median_NDVI_Obs_Over_0.4'] < NDVI_OBS_THRESHOLD]
    
    # 3. Calculate the metrics
    total_farms_rejected_nci = len(rejected_nci_df)
    total_farms_rejected_both = len(rejected_both_df)
    
    # --- Display Metrics ---
    st.subheader(f"NCI and Vegetation History Filter (Median NDVI Obs > 0.4 < {NDVI_OBS_THRESHOLD})")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    col_s1.metric(
        "Total Farms Rejected by NCI Score",
        total_farms_rejected_nci
    )
    
    col_s2.metric(
        f"Farms Rejected by NCI **AND** Median NDVI Obs < {NDVI_OBS_THRESHOLD}",
        total_farms_rejected_both
    )
    
    # Calculate percentage rejected by both (useful if NCI rejected farms are the universe)
    if total_farms_rejected_nci > 0:
        pct_rejected_both = round((total_farms_rejected_both / total_farms_rejected_nci) * 100, 1)
        col_s3.metric(
            f"% of NCI-Rejected Farms Lacking History (Median < {NDVI_OBS_THRESHOLD} obs)",
            f"{pct_rejected_both}%"
        )
    else:
        col_s3.metric(
            f"% of NCI-Rejected Farms Lacking History (Median < {NDVI_OBS_THRESHOLD} obs)",
            "N/A"
        )
    
    st.dataframe(
        rejected_both_df[['Name', 'Area_Ha', 'BIOME_NAME', 'NCI_Score_Rounded', 'NCI_Threshold', 'Median_NDVI_Obs_Over_0.4']],
        use_container_width=True
    )
    st.markdown("---")
    
# --- SECTION 4: Integrated NCI and NDVI Analysis (Plotting) ---

if is_nci_data_ready and is_ndvi_data_ready:
    st.header("4. Integrated NCI & Vegetation pattern Analysis")
    
    # 1. Merge the two processed dataframes on the common key (using left merge for detailed analysis)
    # Note: processed_ndvi_df still contains the multiple rows per kyari_id here! 
    # This is fine for merging the NDVI columns (ndvi_cols) which are needed for plotting.
    merged_df = pd.merge(
        nci_df, 
        processed_ndvi_df[['Primary_Key', 'Num_Cycles','Total_Cycles_Summary', 'Exception_Cycles_Count', 'NDVI_Obs_Over_0.4', 'Detailed_Cycles'] + ndvi_cols],
        left_on='Primary_Key', 
        right_on='Primary_Key',
        how='left', 
        suffixes=('_NCI', '_NDVI')
    )
    
    # Handle duplicates created by the merge (if one NCI field links to multiple NDVI rows), 
    # keep the first instance for plotting and detail view.
    merged_df = merged_df.drop_duplicates(subset=['Primary_Key'], keep='first')
    
    merged_df = merged_df.drop(columns=['Primary_Key'], errors='ignore')

    # --- FILTERING STEP ---
    st.subheader("Field-Level Detail Selection")
    
    col_filter_nci, col_filter_select = st.columns([1, 2])
    
    # 2. Status Label Filter (Requested Filter)
    nci_status_options = merged_df['Status_Label'].unique().tolist()
    nci_status_options.insert(0, 'All Statuses') 
    
    selected_nci_status = col_filter_nci.selectbox(
        "Filter by NCI Status:",
        nci_status_options
    )
    
    # Apply status filter
    if selected_nci_status != 'All Statuses':
        filtered_df = merged_df[merged_df['Status_Label'] == selected_nci_status]
    else:
        filtered_df = merged_df.copy()
    
    # 3. Key Selection Box
    available_keys = filtered_df['Name'].dropna().unique()
    
    if available_keys.size > 0:
        
        selected_key = col_filter_select.selectbox(
            "Select Field/Kyari ID for Detailed Analysis:",
            available_keys,
            index=0 
        )
        
        # 4. Filter the data for the selected key
        selected_row = filtered_df[filtered_df['Name'] == selected_key].iloc[0]
        
        st.subheader(f"Farm level historical vegetation pattern analysis for **{selected_key}**")
        
        def safe_int_conversion(value):
            if pd.isna(value) or value is None:
                return 0
            return int(value)
            
        st.markdown("""
        <style>
        /* Target the metric label (the title) */
        [data-testid="stMetricLabel"] > div {
            font-size: 0.8em; /* 80% of normal size */
        }
        /* Target the metric value (the main number) */
        [data-testid="stMetricValue"] {
            font-size: 1.2em; /* Slightly smaller main value */
        }
        </style>
        """, unsafe_allow_html=True)

        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        
        col_d1.metric("NCI Status", selected_row['Status_Label'])
        col_d2.metric("NCI Score", f"{selected_row['NCI_Score_Rounded']} (Threshold: {selected_row['NCI_Threshold']})")
        
        col_d3.metric("Total Valid Cycles", selected_row['Total_Cycles_Summary'])
        col_d4.metric("NDVI Obs > 0.4", safe_int_conversion(selected_row['NDVI_Obs_Over_0.4']))
        
        st.markdown("---")
        
        # 5. Plot the NDVI Time Series Graph
        st.subheader("NDVI Crop Cycle Time Series Plot")
        
        if 'Detailed_Cycles' in selected_row:
            ndvi_fig = plot_crop_cycle_analysis(
                selected_row, 
                ndvi_cols, 
                ndvi_start_threshold=NDVI_START_THRESHOLD, 
                ndvi_end_threshold=NDVI_END_THRESHOLD
            )
            st.pyplot(ndvi_fig)
        else:
            st.warning(f"NDVI time series data for key '{selected_key}' is missing or incomplete.")
        
    else:
        if selected_nci_status != 'All Statuses':
            st.warning(f"No records found with NCI Status: **{selected_nci_status}** in the filtered list.")
        else:
            st.warning("No field records found after merging NCI and NDVI data.")


elif nci_uploaded_file is not None and ndvi_uploaded_file is None:
    st.info("NDVI data not uploaded. Please upload the NDVI file to enable the integrated analysis (Section 3 and 4).")
    
elif nci_uploaded_file is None:
    st.info("Please upload your NCI/Area data file using the panel on the left sidebar to start the analysis.")

if is_nci_data_ready:
    # 3. DATA LOADED SUCCESSFULLY
    st.header("NCI Data")
    st.success(f"Successfully loaded and processed {len(nci_df)} records from the uploaded NCI file.")
    st.dataframe(nci_df[['BIOME_NAME', 'District', 'Name', 'State', 'Area_Ha', 'NCI_Score', 'NCI_Threshold', 'Status_Label']], use_container_width=True)
    st.markdown("---")
