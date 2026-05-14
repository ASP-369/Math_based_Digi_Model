from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from src.twin.components import IntakeState, TwinInputs
from src.twin.materials import MEMBRANE_LIBRARY
from src.twin.ml_client import ExternalModelConfig, call_external_model
from src.twin.plant import simulate_desalination_plant


st.set_page_config(page_title="Desalination Math Twin", layout="wide")
st.title("RO + Desalination Mathematical Twin")
st.caption("Interactive local digital twin with clean integration hooks for external ML models.")

with st.sidebar:
    st.header("Feed & Operating Conditions")
    salinity = st.slider("Feed salinity (g/L)", 5.0, 50.0, 35.0, 0.5)
    temperature = st.slider("Feed temperature (C)", 5.0, 45.0, 25.0, 0.5)
    feed_flow = st.slider("Feed flow (m3/h)", 5.0, 200.0, 100.0, 1.0)

    st.header("RO Train")
    pressure = st.slider("RO pressure (bar)", 10.0, 90.0, 60.0, 0.5)
    membrane_area = st.slider("Membrane area (m2)", 10.0, 5000.0, 1500.0, 10.0)
    recovery = st.slider("Recovery target", 0.1, 0.8, 0.45, 0.01)
    pretreatment_loss = st.slider("Pretreatment pressure loss (bar)", 0.1, 10.0, 2.0, 0.1)

    st.header("Energy")
    pump_eff = st.slider("Pump efficiency", 0.3, 0.95, 0.82, 0.01)
    erd_eff = st.slider("Energy recovery efficiency", 0.0, 0.95, 0.65, 0.01)

    st.header("Membrane Material")
    membrane_name = st.selectbox("Material", list(MEMBRANE_LIBRARY.keys()))

material = MEMBRANE_LIBRARY[membrane_name]
inputs = TwinInputs(
    intake=IntakeState(
        salinity_g_per_l=salinity,
        temperature_c=temperature,
        feed_flow_m3_per_h=feed_flow,
    ),
    pressure_bar=pressure,
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
            f"Salinity {salinity:.1f} g/L, Temp {temperature:.1f} C",
            f"Loss {pretreatment_loss:.1f} bar",
            f"Efficiency {pump_eff:.2f}",
            f"{membrane_name}, Rejection {result.rejection_fraction:.4f}",
            f"Efficiency {erd_eff:.2f}",
            f"Permeate salinity {result.permeate_salinity_g_per_l:.3f} g/L",
        ],
    }
)
st.dataframe(component_df, use_container_width=True, hide_index=True)

st.subheader("External ML Endpoint Hook")
use_external_ml = st.toggle("Call external cloud ML model")
if use_external_ml:
    endpoint_url = st.text_input("Endpoint URL", value=os.getenv("ML_ENDPOINT_URL", ""))
    api_key = st.text_input("API key (optional)", value=os.getenv("ML_API_KEY", ""), type="password")

    payload = {
        "twin_inputs": {
            "salinity_g_per_l": salinity,
            "temperature_c": temperature,
            "feed_flow_m3_per_h": feed_flow,
            "pressure_bar": pressure,
            "membrane_area_m2": membrane_area,
            "recovery_target": recovery,
            "pump_efficiency": pump_eff,
            "energy_recovery_efficiency": erd_eff,
            "material": membrane_name,
        },
        "twin_outputs": {
            "specific_energy_kwh_per_m3": result.specific_energy_kwh_per_m3,
            "permeate_salinity_g_per_l": result.permeate_salinity_g_per_l,
            "membrane_stress_index": result.membrane_stress_index,
        },
    }

    if st.button("Run ML prediction"):
        if not endpoint_url:
            st.error("Please provide the endpoint URL.")
        else:
            try:
                cfg = ExternalModelConfig(endpoint_url=endpoint_url, api_key=api_key or None)
                ml_response = call_external_model(cfg, payload)
                st.success("ML endpoint called successfully.")
                st.json(ml_response)
            except Exception as exc:
                st.error(f"ML endpoint call failed: {exc}")

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
