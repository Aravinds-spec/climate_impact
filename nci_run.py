import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

# Define the Biome Thresholds globally (NCI thresholds for rejection)
BIOME_THRESHOLDS = {
    'Tropical & Subtropical Dry Broadleaf Forests': 18.5,
    'Tropical & Subtropical Moist Broadleaf Forests': 8.5,
    'Deserts & Xeric Shrublands': 39.5
}

# --- Data Preparation Functions ---

@st.cache_data
def load_data(uploaded_file):
    """Loads and preprocesses the DataFrame."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        # Fallback for manual parsing
        if hasattr(uploaded_file, 'getvalue'):
             try:
                 df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')), sep='|')
             except Exception:
                 st.error("Failed to read file as CSV or custom-separated format.")
                 return pd.DataFrame()
        else:
            return pd.DataFrame()

    # --- CRITICAL: Column Assignment based on Sample Data ---
    numerical_cols = df.select_dtypes(include=np.number).columns
    
    if len(numerical_cols) >= 2:
        df['Area_Ha'] = df[numerical_cols[-2]]
        df['NCI_Score'] = df[numerical_cols[-1]]
    else:
        st.error("Could not identify the two required numerical columns (Area_Ha and NCI_Score). Please check your file.")
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

# --- Summary and Plotting Functions ---

def get_overall_rejection_summary(data_df):
    """Calculates the overall summary focused on NCI Rejection, now using area for the rejection rate."""
    rejected_df = data_df[data_df['is_rejected'] == True]
    accepted_df = data_df[data_df['is_rejected'] == False]
    
    total_records = len(data_df)
    total_rejected = len(rejected_df)
    total_accepted = len(accepted_df)
    
    # --- Robust Area Calculation ---
    total_area_rejected = round(rejected_df['Area_Ha'].sum(), 2) if not rejected_df.empty else 0.0
    total_area_accepted = round(accepted_df['Area_Ha'].sum(), 2) if not accepted_df.empty else 0.0
    total_farm_area = round(data_df['Area_Ha'].sum(), 2)
    
    avg_nci_rejected = round(rejected_df['NCI_Score'].mean(), 2) if not rejected_df.empty else 0.0

    # --- Area-Based Rejection Rate Calculation ---
    area_rejection_rate = 0
    if total_farm_area > 0:
        area_rejection_rate = round((total_area_rejected / total_farm_area) * 100, 2)

    # --- Construct Summary Dictionary ---

    overall_summary = {
        'Total Records': total_records,
        'Records Rejected': total_rejected,
        'Records Accepted': total_accepted,
        
        'Total Area (Ha) Under Analysis': total_farm_area,
        
        # CHANGED: This is the Area Rejection Rate
        'Area Rejection Rate (%)': area_rejection_rate, 
        
        'Total Area Rejected (Ha)': total_area_rejected,
        'Total Area Accepted (Ha)': total_area_accepted,
        
        'Avg NCI Score (Rejected)': avg_nci_rejected,
    }
    return overall_summary

def get_grouped_status_summary(data_df, group_col):
    """Calculates acceptance/rejection count and area summary by a grouping column (Biome/State)."""
    
    REJECTED_LABEL = 'REJECTED (NCI < Threshold)'
    ACCEPTED_LABEL = 'ACCEPTED (NCI >= Threshold)'
    
    # 1. Group and Unstack (Creates MultiIndex Columns)
    # The columns will be ('Count', 'REJECTED LABEL'), ('Total_Area_Ha', 'REJECTED LABEL'), etc.
    summary = data_df.groupby([group_col, 'Status_Label']).agg(
        Count=('Name', 'count'), 
        Total_Area_Ha=('Area_Ha', 'sum')
    ).unstack(fill_value=0)
    
    # --- 2. Handle Missing Status Columns and Ensure Order ---
    required_cols = [
        ('Count', REJECTED_LABEL), ('Count', ACCEPTED_LABEL), 
        ('Total_Area_Ha', REJECTED_LABEL), ('Total_Area_Ha', ACCEPTED_LABEL)
    ]
    
    for col in required_cols:
        if col not in summary.columns:
            summary[col] = 0
    
    # --- 3. FINAL FIX: Create a simple, single-level Index from the MultiIndex components ---
    
    # Define the desired final names in the correct order
    final_names = [
        'Rejected_Count', 'Accepted_Count', 
        'Rejected_Area_Ha', 'Accepted_Area_Ha'
    ]
    
    # Define the order of the source columns (tuples)
    source_cols = [
        ('Count', REJECTED_LABEL), ('Count', ACCEPTED_LABEL),
        ('Total_Area_Ha', REJECTED_LABEL), ('Total_Area_Ha', ACCEPTED_LABEL)
    ]
    
    # Filter the DataFrame to ONLY the required source columns
    summary = summary[source_cols]
    
    # Assign the new, single-level index (this definitively flattens and names the columns)
    summary.columns = final_names
    
    # --- 4. Final Calculations ---
    
    # The columns are now simple strings, so this indexing works perfectly
    # summary = summary[['Rejected_Count', 'Accepted_Count', 'Rejected_Area_Ha', 'Accepted_Area_Ha']] 
    # (The previous line is no longer necessary as we already filtered/assigned in step 3)
    
    summary['Total_Records'] = summary['Rejected_Count'] + summary['Accepted_Count']
    summary['Total_Area_Ha'] = summary['Rejected_Area_Ha'] + summary['Accepted_Area_Ha']
    
    # Area-Based Rejection Rate
    summary['Area_Rejection_Rate_%'] = (summary['Rejected_Area_Ha'] / summary['Total_Area_Ha'] * 100).round(1).fillna(0)

    # Return the required columns
    return summary[['Total_Records', 'Total_Area_Ha', 'Rejected_Area_Ha', 
                    'Accepted_Area_Ha', 'Area_Rejection_Rate_%']].sort_values(by='Rejected_Area_Ha', ascending=False)

# --- STREAMLIT APP LAYOUT ---

st.set_page_config(layout="wide", page_title="NCI Rejection Analysis Dashboard", initial_sidebar_state="expanded")

st.title("🔴 NCI Rejection Analysis Dashboard")
st.markdown("### Focused on Negative Climate Impact (NCI < Threshold) Status")

# File Uploader Section
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your Farm Data CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    if not df.empty:
        # --- Run Analysis and Display ---

        # 1. OVERALL NCI REJECTION SUMMARY (First Focus)
        overall_summary = get_overall_rejection_summary(df)
        
        st.header("1. Overall NCI Status Summary")
        col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
        
        col_m1.metric("Total Records Analyzed", overall_summary['Total Records'])
        col_m2.metric("Total Area (Ha) Under Analysis", overall_summary['Total Area (Ha) Under Analysis'])
        col_m3.metric("Total Records Rejected", overall_summary['Records Rejected'])
        
        # CHANGED: Displaying Area Rejection Rate
        col_m4.metric("Area Rejection Rate (%)", f"{overall_summary['Area Rejection Rate (%)']}%") 
        
        col_m5.metric("Total Area Rejected (Ha)", overall_summary['Total Area Rejected (Ha)'])
        col_m6.metric("Total Area Accepted (Ha)", overall_summary['Total Area Accepted (Ha)'])
        
        # Plot: Overall Rejection vs Acceptance Area (Changed to Area-based Pie Chart)
        overall_status_df = pd.DataFrame({
            'Status': ['REJECTED AREA', 'ACCEPTED AREA'],
            'Area_Ha': [overall_summary['Total Area Rejected (Ha)'], overall_summary['Total Area Accepted (Ha)']]
        })
        fig_overall = px.pie(
            overall_status_df, 
            values='Area_Ha', 
            names='Status', 
            title='Overall Area Distribution: Rejected vs. Accepted',
            color='Status',
            color_discrete_map={'REJECTED AREA': 'red', 'ACCEPTED AREA': 'blue'}
        )
        st.plotly_chart(fig_overall, use_container_width=True)
        st.markdown("---")


        # 2. BIOME-BASED REJECTION ANALYSIS
        st.header("2. Biome-Name Based Rejection Analysis")
        biome_status_df = get_grouped_status_summary(df, 'BIOME_NAME')
        
        col_b1, col_b2 = st.columns([1, 1])
        
        with col_b1:
            st.subheader("Area Summary by Biome")
            # Table now shows Total Area, Rejected Area, Accepted Area, and Area Rejection Rate %
            st.dataframe(biome_status_df, use_container_width=True)
            
        with col_b2:
            st.subheader("Area Status Distribution by Biome")
            fig_biome_area = px.bar(
                df,
                x='BIOME_NAME',
                y='Area_Ha',
                color='Status_Label',
                title='Area (Ha) Rejected vs. Accepted by Biome',
                hover_name='District',
                color_discrete_map={'REJECTED (NCI < Threshold)': 'orange', 'ACCEPTED (NCI >= Threshold)': 'blue'}
            )
            fig_biome_area.update_layout(xaxis={'categoryorder': 'total descending'}, yaxis_title="Total Area (Ha)")
            st.plotly_chart(fig_biome_area, use_container_width=True)
        st.markdown("---")


        # 3. STATE-BASED REJECTION ANALYSIS
        st.header("3. State-Based Rejection Analysis")
        state_status_df = get_grouped_status_summary(df, 'State')
        
        col_s1, col_s2 = st.columns([1, 1])

        with col_s1:
            st.subheader("Area Summary by State")
            # Table now shows Total Area, Rejected Area, Accepted Area, and Area Rejection Rate %
            st.dataframe(state_status_df, use_container_width=True)
            
        with col_s2:
            st.subheader("Count Status Distribution by State")
            
            # Use a Count Bar Plot (as requested)
            df['Record_Count'] = 1 
            
            fig_state_count = px.bar(
                df,
                x='State',
                y='Record_Count',  # Y-axis is the count
                color='Status_Label',
                title='Record Count Rejected vs. Accepted by State',
                hover_name='District',
                color_discrete_map={
                    'REJECTED (NCI < Threshold)': 'orange', 
                    'ACCEPTED (NCI >= Threshold)': 'blue'
                }
            )
            fig_state_count.update_layout(
                xaxis={'categoryorder': 'total descending'},
                yaxis_title="Number of Records"
            )
            
            st.plotly_chart(fig_state_count, use_container_width=True)
            del df['Record_Count'] # Clean up temporary column

        st.markdown("---")

        
        # 4. DATA LOADED SUCCESSFULLY (Last Requirement)
        st.header("4. Data Loaded Successfully")
        st.success(f"Successfully loaded and processed {len(df)} records from the uploaded file.")
        st.dataframe(df[['BIOME_NAME', 'District', 'Name', 'State', 'Area_Ha', 'NCI_Score', 'NCI_Threshold', 'Status_Label']], use_container_width=True)
        
    else:
        st.warning("Please check your uploaded CSV file. It might be empty, formatted incorrectly, or missing key columns.")

else:
    st.info("Please upload your data file using the panel on the left sidebar to start the NCI Rejection Analysis.")
