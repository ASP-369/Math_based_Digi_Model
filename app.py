import os
from datetime import datetime
import json
import requests
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import streamlit as st

from src.twin.components import IntakeState, TwinInputs
from src.twin.materials import MEMBRANE_LIBRARY
from src.twin.ml_client import ExternalModelConfig, call_external_model
from src.twin.plant import simulate_desalination_plant

# Load environment variables from .env
load_dotenv()

# Azure ML Endpoint Constants
ELECTRICITY_URI = os.getenv("ELECTRICITY_URI")
MEMBRANE_URI = os.getenv("MEMBRANE_URI")
ELECTRICITY_KEY = os.getenv("ELECTRICITY_KEY")
MEMBRANE_KEY = os.getenv("MEMBRANE_KEY")

base_headers_elec = {"Content-Type": "application/json"}
base_headers_mem = {"Content-Type": "application/json"}

if ELECTRICITY_KEY:
    base_headers_elec["Authorization"] = f"Bearer {ELECTRICITY_KEY}"

if MEMBRANE_KEY:
    base_headers_mem["Authorization"] = f"Bearer {MEMBRANE_KEY}"

st.set_page_config(page_title="Desalination Math Twin", layout="wide")
st.title("RO + Desalination Mathematical Twin")
st.caption("Interactive local digital twin with clean integration hooks for external ML models.")

with st.sidebar:
    st.header("Feed & Operating Conditions")
    
    st.write("**feed_conductivity_uscm**")
    c1, c2 = st.columns([2, 1])
    feed_conductivity_uscm = c1.slider("cond_s", 100.0, 10000.0, 1775.0, 50.0, label_visibility="collapsed")
    feed_conductivity_uscm = c2.number_input("cond_n", 100.0, 10000.0, feed_conductivity_uscm, 50.0, label_visibility="collapsed")

    st.write("**Feed temperature (C)**")
    c1, c2 = st.columns([2, 1])
    temperature = c1.slider("temp_s", 5.0, 45.0, 25.0, 0.5, label_visibility="collapsed")
    temperature = c2.number_input("temp_n", 5.0, 45.0, temperature, 0.5, label_visibility="collapsed")

    st.write("**Feed flow (m3/h)**")
    c1, c2 = st.columns([2, 1])
    feed_flow = c1.slider("flow_s", 100.0, 5000.0, 3500.0, 10.0, label_visibility="collapsed")
    feed_flow = c2.number_input("flow_n", 100.0, 5000.0, feed_flow, 10.0, label_visibility="collapsed")

    st.header("RO Train")
    st.write("**system_pressure_mpa**")
    c1, c2 = st.columns([2, 1])
    system_pressure_mpa = c1.slider("press_s", 0.1, 10.0, 0.68, 0.01, label_visibility="collapsed")
    system_pressure_mpa = c2.number_input("press_n", 0.1, 10.0, system_pressure_mpa, 0.01, label_visibility="collapsed", format="%.2f")

    st.write("**Membrane area (m2)**")
    c1, c2 = st.columns([2, 1])
    membrane_area = c1.slider("area_s", 1000.0, 100000.0, 25000.0, 500.0, label_visibility="collapsed")
    membrane_area = c2.number_input("area_n", 1000.0, 100000.0, membrane_area, 500.0, label_visibility="collapsed")

    st.write("**Recovery target**")
    c1, c2 = st.columns([2, 1])
    recovery = c1.slider("rec_s", 0.1, 0.9, 0.50, 0.01, label_visibility="collapsed")
    recovery = c2.number_input("rec_n", 0.1, 0.9, recovery, 0.01, label_visibility="collapsed")

    st.write("**Pretreatment loss (bar)**")
    c1, c2 = st.columns([2, 1])
    pretreatment_loss = c1.slider("loss_s", 0.1, 10.0, 0.5, 0.1, label_visibility="collapsed")
    pretreatment_loss = c2.number_input("loss_n", 0.1, 10.0, pretreatment_loss, 0.1, label_visibility="collapsed")

    st.header("Energy")
    st.write("**pump_power_kw**")
    c1, c2 = st.columns([2, 1])
    pump_power_kw = c1.slider("pow_s", 10.0, 1000.0, 438.0, 1.0, label_visibility="collapsed")
    pump_power_kw = c2.number_input("pow_n", 10.0, 1000.0, pump_power_kw, 1.0, label_visibility="collapsed")

    st.write("**Pump efficiency**")
    c1, c2 = st.columns([2, 1])
    pump_eff = c1.slider("eff_s", 0.3, 0.95, 0.82, 0.01, label_visibility="collapsed")
    pump_eff = c2.number_input("eff_n", 0.3, 0.95, pump_eff, 0.01, label_visibility="collapsed")

    st.write("**Energy recovery efficiency**")
    c1, c2 = st.columns([2, 1])
    erd_eff = c1.slider("erd_s", 0.0, 0.95, 0.65, 0.01, label_visibility="collapsed")
    erd_eff = c2.number_input("erd_n", 0.0, 0.95, erd_eff, 0.01, label_visibility="collapsed")

    st.header("Membrane Material & State")
    membrane_name = st.selectbox("Material", list(MEMBRANE_LIBRARY.keys()))
    
    st.write("**membrane_age_years**")
    c1, c2 = st.columns([2, 1])
    membrane_age_years = c1.slider("age_s", 0.0, 10.0, 0.5, 0.1, label_visibility="collapsed")
    membrane_age_years = c2.number_input("age_n", 0.0, 10.0, membrane_age_years, 0.1, label_visibility="collapsed")

    st.write("**fouling_stage**")
    c1, c2 = st.columns([2, 1])
    fouling_stage = c1.slider("foul_s", 0, 5, 1, label_visibility="collapsed")
    fouling_stage = c2.number_input("foul_n", 0, 5, fouling_stage, 1, label_visibility="collapsed")

    st.header("External ML Features")
    
    st.write("**permeate_flowrate_m3h**")
    c1, c2 = st.columns([2, 1])
    # Default to 1738 as per your example
    permeate_flowrate_m3h = c1.slider("perm_flow_s", 100.0, 5000.0, 1738.0, 10.0, label_visibility="collapsed")
    permeate_flowrate_m3h = c2.number_input("perm_flow_n", 100.0, 5000.0, permeate_flowrate_m3h, 10.0, label_visibility="collapsed")

    st.write("**differential_pressure_mpa**")
    c1, c2 = st.columns([2, 1])
    differential_pressure_mpa = c1.slider("diff_s", 0.01, 0.5, 0.05, 0.005, label_visibility="collapsed")
    differential_pressure_mpa = c2.number_input("diff_n", 0.01, 0.5, differential_pressure_mpa, 0.005, label_visibility="collapsed", format="%.3f")
    
    st.subheader("Time & Tariff")
    now = datetime.now()
    
    st.write("**hour_of_day**")
    c1, c2 = st.columns([2, 1])
    hour_of_day = c1.slider("hour_s", 0, 23, now.hour, label_visibility="collapsed")
    hour_of_day = c2.number_input("hour_n", 0, 23, hour_of_day, 1, label_visibility="collapsed")

    st.write("**month**")
    c1, c2 = st.columns([2, 1])
    month = c1.slider("month_s", 1, 12, now.month, label_visibility="collapsed")
    month = c2.number_input("month_n", 1, 12, month, 1, label_visibility="collapsed")

    is_peak_tariff = st.checkbox("is_peak_tariff", value=(10 <= hour_of_day <= 18))
    is_weekend = st.checkbox("is_weekend", value=(now.weekday() >= 5))
    
    st.write("**electricity_tariff_eur_kwh**")
    electricity_tariff_eur_kwh = st.number_input("electricity_tariff_eur_kwh", 0.0, 1.0, 0.032, 0.001, format="%.3f")

# Convert conductivity back to approximate g/L for the math twin logic
salinity_approx = feed_conductivity_uscm / 1500.0

material = MEMBRANE_LIBRARY[membrane_name]
inputs = TwinInputs(
    intake=IntakeState(
        salinity_g_per_l=salinity_approx,
        temperature_c=temperature,
        feed_flow_m3_per_h=feed_flow,
    ),
    pressure_bar=system_pressure_mpa * 10.0,
    pump_efficiency=pump_eff,
    membrane_area_m2=membrane_area,
    recovery_target=recovery,
    pretreatment_loss_bar=pretreatment_loss,
    energy_recovery_efficiency=erd_eff,
)

result = simulate_desalination_plant(inputs, material)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Permeate flow (m3/h)", f"{result.permeate_flow_m3_per_h:.2f}")
col2.metric("Permeate salinity (g/L)", f"{result.permeate_salinity_g_per_l:.3f}")
col3.metric("Specific energy (kWh/m3)", f"{result.specific_energy_kwh_per_m3:.2f}")
col4.metric("Membrane stress index", f"{result.membrane_stress_index:.2f}")

if result.warnings:
    st.warning(" | ".join(result.warnings))

flow_df = pd.DataFrame(
    {
        "Stream": ["Feed", "Permeate", "Brine"],
        "Flow_m3_h": [feed_flow, result.permeate_flow_m3_per_h, result.brine_flow_m3_per_h],
    }
)

power_df = pd.DataFrame(
    {
        "Metric": ["Net pump power (kW)", "Specific energy (kWh/m3)", "Feed osmotic pressure (bar)", "Net driving pressure (bar)"],
        "Value": [
            result.pump_power_kw,
            result.specific_energy_kwh_per_m3,
            result.feed_osmotic_pressure_bar,
            result.net_driving_pressure_bar,
        ],
    }
)

left, right = st.columns(2)
with left:
    st.subheader("Plant Streams")
    st.plotly_chart(px.bar(flow_df, x="Stream", y="Flow_m3_h", color="Stream"), use_container_width=True)

with right:
    st.subheader("Energy and Pressure")
    st.plotly_chart(px.bar(power_df, x="Metric", y="Value", color="Metric"), use_container_width=True)

st.subheader("Component View")
component_df = pd.DataFrame(
    {
        "Component": [
            "Intake",
            "Pretreatment",
            "High Pressure Pump",
            "RO Membrane",
            "Energy Recovery Device",
            "Post-Treatment",
        ],
        "Status": [
            f"Cond. {feed_conductivity_uscm:.0f} uS/cm, Temp {temperature:.1f} C",
            f"Loss {pretreatment_loss:.1f} bar",
            f"Efficiency {pump_eff:.2f}",
            f"{membrane_name}, Rejection {result.rejection_fraction:.4f}",
            f"Efficiency {erd_eff:.2f}",
            f"Permeate salinity {result.permeate_salinity_g_per_l:.3f} g/L",
        ],
    }
)
st.dataframe(component_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("🚀 Azure ML Cloud Predictions")

tabs = st.tabs(["⚡ Energy Cost Prediction", "🧬 Membrane Health Prediction"])

with tabs[0]:
    st.markdown("### Electricity Cost Predictor (MDN)")
    st.info("Predicts hourly electricity cost based on current operating conditions and time of day.")
    
    # Prepare payload for Energy Model
    energy_payload = {
        "data": [{
            "system_pressure_mpa": system_pressure_mpa,
            "permeate_flowrate_m3h": float(permeate_flowrate_m3h),
            "pump_power_kw": pump_power_kw,
            "feed_conductivity_uscm": float(feed_conductivity_uscm),
            "differential_pressure_mpa": differential_pressure_mpa,
            "hour_of_day": hour_of_day,
            "is_peak_tariff": 1 if is_peak_tariff else 0,
            "month": month,
            "is_weekend": 1 if is_weekend else 0,
            "electricity_tariff_eur_kwh": electricity_tariff_eur_kwh,
            "membrane_age_years": membrane_age_years,
            "fouling_stage": fouling_stage
        }]
    }

    if st.button("Predict Energy Cost", key="btn_energy"):
        try:
            with st.spinner("Calling Energy Model..."):
                print(energy_payload)
                resp = requests.post(ELECTRICITY_URI, json=energy_payload, headers=base_headers_elec, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    # Handle double-encoded JSON strings from Azure ML
                    if isinstance(data, str):
                        data = json.loads(data)
                    
                    pred = data.get("predictions", [0])[0]
                    st.metric("Predicted Hourly Cost", f"€ {pred:.4f}")
                    st.success("Prediction received from Azure ML.")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Failed to connect to endpoint: {e}")

with tabs[1]:
    st.markdown("### Membrane Health LSTM (24h Sequence)")
    st.info("Uses the past 24 hours of data to predict the current health score.")
    
    # Sample sequence for demonstration as requested by user
    sample_seq = [
        [1690.0, 0.67, 0.048, 98.7, 1.01, 1785.0, 25.5, 0.48, 1],
        [1691.5, 0.67, 0.048, 98.7, 1.01, 1786.0, 25.4, 0.48, 1],
        [1693.0, 0.67, 0.048, 98.7, 1.01, 1787.0, 25.3, 0.48, 1],
        [1694.2, 0.67, 0.048, 98.7, 1.01, 1788.0, 25.3, 0.48, 1],
        [1695.0, 0.67, 0.048, 98.7, 1.01, 1789.0, 25.2, 0.48, 1],
        [1696.3, 0.67, 0.048, 98.7, 1.01, 1790.0, 25.2, 0.48, 1],
        [1697.8, 0.68, 0.049, 98.6, 1.01, 1791.0, 25.1, 0.48, 1],
        [1699.1, 0.68, 0.049, 98.6, 1.01, 1792.0, 25.1, 0.48, 1],
        [1700.0, 0.68, 0.049, 98.6, 1.01, 1793.0, 25.0, 0.49, 1],
        [1701.2, 0.68, 0.049, 98.6, 1.01, 1794.0, 25.0, 0.49, 1],
        [1702.5, 0.68, 0.050, 98.6, 1.02, 1795.0, 24.9, 0.49, 1],
        [1703.9, 0.68, 0.050, 98.6, 1.02, 1796.0, 24.9, 0.49, 1],
        [1705.0, 0.68, 0.050, 98.5, 1.02, 1797.0, 24.8, 0.49, 1],
        [1706.1, 0.68, 0.050, 98.5, 1.02, 1798.0, 24.8, 0.49, 1],
        [1707.0, 0.68, 0.050, 98.5, 1.02, 1799.0, 24.7, 0.49, 1],
        [1708.3, 0.68, 0.051, 98.5, 1.02, 1800.0, 24.7, 0.50, 1],
        [1709.4, 0.68, 0.051, 98.5, 1.02, 1801.0, 24.6, 0.50, 1],
        [1710.0, 0.68, 0.051, 98.4, 1.02, 1802.0, 24.6, 0.50, 1],
        [1711.2, 0.68, 0.051, 98.4, 1.02, 1803.0, 24.5, 0.50, 1],
        [1712.5, 0.68, 0.052, 98.4, 1.03, 1804.0, 24.5, 0.50, 1],
        [1713.7, 0.68, 0.052, 98.4, 1.03, 1805.0, 24.4, 0.50, 1],
        [1714.9, 0.68, 0.052, 98.3, 1.03, 1806.0, 24.4, 0.50, 1],
        [1716.0, 0.68, 0.052, 98.3, 1.03, 1807.0, 24.3, 0.50, 1],
        [1717.0, 0.68, 0.052, 98.3, 1.03, 1808.0, 24.3, 0.50, 1]
    ]

    # Current data point based on sliders
    current_point = [
        float(permeate_flowrate_m3h),
        system_pressure_mpa,
        differential_pressure_mpa,
        result.rejection_fraction * 100.0,
        result.net_driving_pressure_bar / 10.0, # Approximate normalized flux
        float(feed_conductivity_uscm),
        float(feed_conductivity_uscm) * 0.02, # Approximate permeate conductivity
        membrane_age_years,
        fouling_stage
    ]

    # Mix current state into sample for "real-time" feel
    dynamic_seq = sample_seq[:-1] + [current_point]
    
    membrane_payload = {"data": [dynamic_seq]}

    if st.button("Analyze Membrane Health", key="btn_membrane"):
        try:
            with st.spinner("Calling LSTM Model..."):
                resp = requests.post(MEMBRANE_URI, json=membrane_payload, headers=base_headers_mem, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    # Handle double-encoded JSON strings from Azure ML
                    if isinstance(data, str):
                        data = json.loads(data)
                        
                    health_score = data.get("predictions", [0])[0]
                    st.metric("Membrane Health Score", f"{health_score:.4f}")
                    if health_score > 0.8:
                        st.success("Health is Optimal.")
                    elif health_score > 0.5:
                        st.warning("Health is Degrading. Monitor closely.")
                    else:
                        st.error("Low Health Score! Maintenance required.")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Failed to connect to endpoint: {e}")

st.subheader("How this twin works")
st.markdown(
    """
- Osmotic pressure is estimated by Van't Hoff relation from salinity and temperature.
- Net driving pressure = applied pressure - pretreatment losses - average osmotic pressure.
- Permeate flux uses a simple solution-diffusion style term with membrane permeability.
- Energy uses hydraulic power with pump and energy-recovery efficiencies.
- Membrane stress index is a compact health indicator for monitoring/fouling risk.
"""
)
