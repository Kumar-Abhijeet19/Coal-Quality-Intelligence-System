import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import plotly.express as px

# Page Configurations
st.set_page_config(page_title="BCCL Coal Analytics", layout="wide", page_icon="🏭")

# Custom Styling for Premium Industrial Theme
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 13px;
        color: #94a3b8;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 24px;
        color: #f8fafc;
        font-weight: bold;
        margin-top: 5px;
    }
    .recommendation-box {
        background-color: #0f172a;
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

MODEL_PATH = "coal_grade_model.pkl"

@st.cache_resource
def load_industrial_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error initializing predictive engine asset: {e}")
    return None

model = load_industrial_model()

# Core BCCL Grade Assignment Logic (G1 - G18)
def assign_grade_local(gcv):
    try:
        gcv_val = float(gcv)
    except (ValueError, TypeError):
        return 'G18' # Fallback for corrupted data rows
        
    if gcv_val >= 7000: return 'G1'
    elif 6700 < gcv_val <= 7000: return 'G2'
    elif 6400 < gcv_val <= 6700: return 'G3'
    elif 6100 < gcv_val <= 6400: return 'G4'
    elif 5800 < gcv_val <= 6100: return 'G5'
    elif 5500 < gcv_val <= 5800: return 'G6'
    elif 5200 < gcv_val <= 5500: return 'G7'
    elif 4900 < gcv_val <= 5200: return 'G8'
    elif 4600 < gcv_val <= 4900: return 'G9'
    elif 4300 < gcv_val <= 4600: return 'G10'
    elif 4000 < gcv_val <= 4300: return 'G11'
    elif 3700 < gcv_val <= 4000: return 'G12'
    elif 3400 < gcv_val <= 3700: return 'G13'
    elif 3100 < gcv_val <= 3400: return 'G14'
    elif 2800 < gcv_val <= 3100: return 'G15'
    elif 2500 < gcv_val <= 2800: return 'G16'
    elif 2200 < gcv_val <= 2500: return 'G17'
    else: return 'G18'

price_power_dict = {
    'G1': 7400, 'G2': 6900, 'G3': 6420, 'G4': 6000, 'G5': 5500,
    'G6': 3800, 'G7': 3200, 'G8': 2840, 'G9': 2200, 'G10': 1960,
    'G11': 1620, 'G12': 1520, 'G13': 1440, 'G14': 1300, 'G15': 1200,
    'G16': 1060, 'G17': 940, 'G18': 700  
}

price_other_dict = {
    'G1': 7400, 'G2': 6900, 'G3': 6420, 'G4': 6000, 'G5': 5500,
    'G6': 4560, 'G7': 3840, 'G8': 3400, 'G9': 2640, 'G10': 2360,
    'G11': 1940, 'G12': 1820, 'G13': 1720, 'G14': 1560, 'G15': 1440,
    'G16': 1280, 'G17': 1140, 'G18': 700  
}

def analyze_coal_parameters(ash, moisture, sulphur, volatile, predicted_gcv, grade_label):
    custom_recs = []
    pipeline_steps = []
    
    # 1. Check for Upstream / Processing Methods (Washing)
    if ash >= 22:
        pipeline_steps.append("Coal Washing")
        custom_recs.append(f"⚠️ **High Ash ({ash}%):** Reroute payload to **Coal Washeries** to separate stone/shale and upgrade GCV.")
    
    # 2. Check for Pre-Combustion Methods (Drying)
    if moisture >= 9:
        pipeline_steps.append("Pre-Drying")
        custom_recs.append(f"⚠️ **High Moisture ({moisture}%):** Initiate pre-combustion thermal drying sequence using waste flue gas heat to prevent mill choking and boiler efficiency drop.")
    
    # 3. Check for Downstream / Exhaust Methods (FGD Loop)
    if sulphur >= 0.8:
        pipeline_steps.append("FGD Loop")
        custom_recs.append(f"⚠️ **High Sulphur ({sulphur}%):** Mandatory routing through **FGD (Flue Gas Desulfurization) Loop** to mitigate SO₂ emissions and comply with environmental norms.")
        
    # --- Industrial End-Use Use Case Routing ---
    # Checking for Metallurgical / Coking Coal Characteristics
    if predicted_gcv >= 6100 and 20 <= volatile <= 32 and ash <= 15:
        use_case = "Premium Metallurgical Coking Coal (Steel Industry / Coke Ovens)"
        pipeline_steps.insert(0, "Direct Carbonization/Coking")
    elif predicted_gcv >= 5500:
        use_case = "High-Grade Commercial Combustion / Cement & Heavy Industries"
    elif predicted_gcv >= 4300:
        use_case = "Power Generation (Commercial & Captive Power Plants)"
    else:
        use_case = "Utility Power Generation (Thermal Power Utilities)"

    # 4. Handle "Multiple Hits" or Empty Pipelines
    if len(pipeline_steps) > 1:
        # Build a visual pipeline chain string (e.g., "Coal Washing ➔ Pre-Drying ➔ FGD Loop")
        pipeline_chain = " ➔ ".join([f"[{step}]" for step in pipeline_steps])
        custom_recs.insert(0, f"⚙️ **Optimized Processing Pipeline Map:** {pipeline_chain}")
    elif not pipeline_steps:
        custom_recs.append("✅ Material characteristics conform to standard operational baselines. No advanced preprocessing required.")

    return custom_recs, use_case

# --- HEADER WITH LOCAL LOGO AND SYSTEM FALLBACK ---
logo_url = "logo.png" 
backup_online_logo = "https://raw.githubusercontent.com/tusharw/all-india-government-logos/master/logos/coal-india-limited.png"

col_logo, col_title = st.columns([1, 10])
with col_logo:
    if os.path.exists(logo_url):
        st.image(logo_url, width=80) 
    else:
        st.image(backup_online_logo, width=80) 

with col_title:
    st.markdown('<h1 style="font-size: 32px; margin-bottom: 0;">BCCL Coal Quality Intelligence System</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="font-size: 18px; margin-top: 0; color: #a3a3a3;">Next-Gen Machine Learning Decision Support System</h3>', unsafe_allow_html=True)

st.markdown(f"**Developed by: Kumar Abhijeet **")
st.write("---")

if model is None:
    st.error(f"❌ Core System Error: Asset `{MODEL_PATH}` not detected. Verification of pipeline configuration required.")
else:
    tab1, tab2 = st.tabs(["⚙️ Core Predictive Engine", "📊 Fleet Logistics Batch Analysis"])

    # --- TAB 1: SINGLE MINE ---
    with tab1:
        col1, col2 = st.columns([4, 5])
        with col1:
            st.subheader("📋 Laboratory Proximate Analysis Input")
            
            st.markdown("##### ⚡ Quick Load from Individual CSV")
            single_csv_file = st.file_uploader("Upload single record CSV to auto-fill", type=["csv"], key="single_uploader")
            
            # Default Baseline Values
            default_mine = "BCCL-Dhanbad-09"
            default_qty = 50.0
            default_carbon = 52.0
            default_volatile = 28.5
            default_ash = 14.2
            default_moisture = 4.8
            default_sulphur = 0.55
            
            if single_csv_file is not None:
                try:
                    loaded_df = pd.read_csv(single_csv_file)
                    if len(loaded_df) > 0:
                        row = loaded_df.iloc[0]
                        default_mine = str(row.get('Mine Terminal ID', row.get('mine_id', default_mine)))
                        default_qty = float(row.get('Quantity (Tonn)', row.get('Quantity', row.get('quantity', default_qty))))
                        default_carbon = float(row.get('Carbon_content (%)', row.get('Carbon_content', row.get('carbon', default_carbon))))
                        default_volatile = float(row.get('Volatile_Matter (%)', row.get('Volatile_Matter', row.get('volatile', default_volatile))))
                        default_ash = float(row.get('Ash_Content (%)', row.get('Ash_Content', row.get('ash', default_ash))))
                        default_moisture = float(row.get('Moisture_content (%)', row.get('Moisture_content', row.get('moisture', default_moisture))))
                        default_sulphur = float(row.get('Sulphur_content (%)', row.get('Sulphur_content', row.get('sulphur', default_sulphur))))
                        
                        st.success("✅ Individual CSV data successfully mapped to the form below!")
                except Exception as e:
                    st.error(f"Error mapping file structure: {e}")
            
            st.write("---")
            
            with st.form("single_prediction"):
                mine_id = st.text_input("Mine Terminal ID", value=default_mine)
                quantity = st.number_input("Coal Quantity (in Tonn)", value=default_qty, step=0.5, min_value=1.0)
                
                st.write("---")
                carbon = st.number_input("Fixed Carbon (FC %)", value=default_carbon, step=0.1)
                volatile = st.number_input("Volatile Matter (VM %)", value=default_volatile, step=0.1)
                ash = st.number_input("Ash Content (A %)", value=default_ash, step=0.1)
                moisture = st.number_input("Total Moisture (TM %)", value=default_moisture, step=0.1)
                sulphur = st.number_input("Sulphur Content (S %)", value=default_sulphur, step=0.01)
                
                submit_btn = st.form_submit_button("Execute ML Inference Pipeline", type="primary")

        with col2:
            st.subheader("📊 Operational Analytics Output")
            if submit_btn:
                features = pd.DataFrame([{
                    'Carbon_content (%)': carbon, 
                    'Volatile_Matter (%)': volatile,
                    'Ash_Content (%)': ash, 
                    'Moisture_content (%)': moisture, 
                    'Sulphur_content (%)': sulphur
                }])
                
                predicted_gcv = float(model.predict(features)[0])
                predicted_grade = assign_grade_local(predicted_gcv)
                
                price_power = price_power_dict.get(predicted_grade, 0)
                price_other = price_other_dict.get(predicted_grade, 0)
                
                total_rev_power = price_power * quantity
                total_rev_other = price_other * quantity
                
                power_display = f"₹ {total_rev_power:,.2f} (@ ₹ {price_power}/Ton)" if price_power > 0 else "Pricing Tier Unregulated"
                other_display = f"₹ {total_rev_other:,.2f} (@ ₹ {price_other}/Ton)" if price_other > 0 else "Pricing Tier Unregulated"

                recs, uses = analyze_coal_parameters(ash, moisture, sulphur, volatile, predicted_gcv, predicted_grade)
                
                st.markdown(f"""
                    <div class="metric-card" style="border-left-color: #3b82f6;"><div class="metric-title">Predicted GCV Matrix</div><div class="metric-value">{predicted_gcv:.1f} kcal/kg</div></div>
                    <div class="metric-card" style="border-left-color: #22c55e;"><div class="metric-title">Assigned Coal Grade</div><div class="metric-value">{predicted_grade}</div></div>
                    <div class="metric-card" style="border-left-color: #eab308;"><div class="metric-title">Power Utilities / Defence Revenue</div><div class="metric-value">{power_display}</div></div>
                    <div class="metric-card" style="border-left-color: #a855f7;"><div class="metric-title">Non-Power Sectors Revenue</div><div class="metric-value">{other_display}</div></div>
                """, unsafe_allow_html=True)
                
                st.info(f"🚚 **Uses:** For {quantity:.2f} Tonns -> {uses}")
                
                st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                st.markdown("#### 🤖 Automated Process Engineering Recommendations:")
                for rec in recs:
                    st.markdown(f"- {rec}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("💡 **Awaiting Laboratory Parameters:** Populate the laboratory proximate data matrices on the left panel to execute inference workflows.")

    # --- TAB 2: BULK ANALYSIS ---
    with tab2:
        st.subheader("📁 Industrial Batch Processing System")
        uploaded_file = st.file_uploader("Upload Analytical Data Manifest (CSV/Excel)", type=["csv", "xlsx"], key="bulk_uploader")
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    bulk_df = pd.read_csv(uploaded_file)
                else:
                    bulk_df = pd.read_excel(uploaded_file)
                
                carbon_col = 'Carbon_content (%)' if 'Carbon_content (%)' in bulk_df.columns else 'Carbon_content'
                volatile_col = 'Volatile_Matter (%)' if 'Volatile_Matter (%)' in bulk_df.columns else 'Volatile_Matter'
                ash_col = 'Ash_Content (%)' if 'Ash_Content (%)' in bulk_df.columns else 'Ash_Content'
                moisture_col = 'Moisture_content (%)' if 'Moisture_content (%)' in bulk_df.columns else 'Moisture_content'
                sulphur_col = 'Sulphur_content (%)' if 'Sulphur_content (%)' in bulk_df.columns else 'Sulphur_content'
                
                qty_col = 'Quantity (Tonn)' if 'Quantity (Tonn)' in bulk_df.columns else ('Quantity' if 'Quantity' in bulk_df.columns else None)
                
                req_found = [carbon_col, volatile_col, ash_col, moisture_col, sulphur_col]
                
                if all(col in bulk_df.columns for col in req_found):
                    if st.button("Initialize Optimization Compute Operations", type="primary"):
                        with st.spinner("Processing batch analytical vector computations..."):
                            
                            bulk_features = pd.DataFrame({
                                'Carbon_content (%)': bulk_df[carbon_col],
                                'Volatile_Matter (%)': bulk_df[volatile_col],
                                'Ash_Content (%)': bulk_df[ash_col],
                                'Moisture_content (%)': bulk_df[moisture_col],
                                'Sulphur_content (%)': bulk_df[sulphur_col]
                            })
                            
                            bulk_gcvs = model.predict(bulk_features)
                            
                            bulk_df['GCV (Kcal/Kg)'] = np.round(bulk_gcvs, 2)
                            
                            # Vectorized assignment using series mapping (Safest for massive datasets)
                            bulk_df['Coal Grade'] = pd.Series(bulk_gcvs).apply(assign_grade_local).values
                            
                            # Base Rate per Ton Mapping
                            bulk_df['Power Utilities/Defence (Rs/Ton)'] = bulk_df['Coal Grade'].map(price_power_dict).fillna(0)
                            bulk_df['Non-Power Sectors (Rs/Ton)'] = bulk_df['Coal Grade'].map(price_other_dict).fillna(0)
                            
                            # Text cleaning string wrapper for quantity field safety
                            if qty_col and qty_col in bulk_df.columns:
                                cleaned_qty = bulk_df[qty_col].astype(str).str.replace(',', '').str.extract(r'(\d+\.?\d*)')[0]
                                total_qty = pd.to_numeric(cleaned_qty, errors='coerce').fillna(1.0)
                            else:
                                total_qty = 1.0
                                st.warning("⚠️ Uploaded sheet me 'Quantity' ya 'Quantity (Tonn)' column nahi mila. Revenue per single ton (1 Ton) ke basis pr calculate kiya gaya hai.")
                            
                            # Total Revenue columns computed accurately
                            bulk_df['Total Revenue: Power Sector (Rs)'] = bulk_df['Power Utilities/Defence (Rs/Ton)'] * total_qty
                            bulk_df['Total Revenue: Non-Power Sector (Rs)'] = bulk_df['Non-Power Sectors (Rs/Ton)'] * total_qty

                            # --- Visual Analytics Section ---
                            st.write("---")
                            st.subheader("📊 Fleet Analytical Insights Dashboard")
                            
                            chart_col, table_col = st.columns([1, 1])
                            
                            with chart_col:
                                grade_counts = bulk_df['Coal Grade'].value_counts().reset_index()
                                grade_counts.columns = ['Grade', 'Count']
                                
                                fig = px.pie(grade_counts, values='Count', names='Grade', 
                                             title="Distribution Profile of Predicted Coal Grades (G1-G18 Categorization)",
                                             hole=0.4, 
                                             color_discrete_sequence=px.colors.qualitative.Bold)
                                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
                                st.plotly_chart(fig, use_container_width=True)

                            with table_col:
                                st.write("### Prediction Manifest Summary Log (Top 10 Batches)")
                                
                                display_cols = [
                                    'GCV (Kcal/Kg)', 
                                    'Coal Grade', 
                                    'Power Utilities/Defence (Rs/Ton)', 
                                    'Total Revenue: Power Sector (Rs)',
                                    'Non-Power Sectors (Rs/Ton)',
                                    'Total Revenue: Non-Power Sector (Rs)'
                                ]
                                
                                # Protected display loop conditional structure
                                custom_meta_cols = [col for col in [qty_col, 'Mine Terminal ID', 'mine_id'] if qty_col is not None and col in bulk_df.columns]
                                final_display = custom_meta_cols + display_cols
                                st.dataframe(bulk_df[final_display].head(10), use_container_width=True)
                            
                            download_df = bulk_df.copy()
                            download_df.replace({0: "Pricing Tier Unregulated"}, inplace=True)
                            csv = download_df.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Export Logistics Revenue Manifest", data=csv, file_name="BCCL_Coal_Quality_And_Revenue_Manifest.csv", mime="text/csv")
                else:
                    st.error(f"❌ Schema Validation Failure! Uploaded document must contain features like Carbon, Ash, Moisture, Sulphur, and Volatile Matter.")
            except Exception as e:
                st.error(f"Pipeline Interruption during asset compute sequence: {e}")