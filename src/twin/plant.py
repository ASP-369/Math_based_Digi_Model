from __future__ import annotations

from .components import SimulationResult, TwinInputs
from .materials import MembraneMaterial


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _osmotic_pressure_bar(salinity_g_per_l: float, temperature_c: float) -> float:
    # Van't Hoff approximation for NaCl-dominant seawater.
    mol_per_l = max(salinity_g_per_l, 0.0) / 58.44
    temperature_k = temperature_c + 273.15
    r = 0.08314  # L bar / (mol K)
    i = 2.0
    return i * mol_per_l * r * temperature_k


def _effective_rejection(material: MembraneMaterial, pressure_bar: float, temperature_c: float) -> float:
    pressure_factor = _clamp(pressure_bar / max(material.max_pressure_bar, 1e-6), 0.0, 1.2)
    temp_penalty = _clamp((temperature_c - 25.0) * 0.0015, -0.02, 0.04)
    rejection = material.nominal_rejection + 0.01 * pressure_factor - temp_penalty
    return _clamp(rejection, 0.94, 0.9995)


def simulate_desalination_plant(inputs: TwinInputs, material: MembraneMaterial) -> SimulationResult:
    warnings: list[str] = []

    feed = inputs.intake
    feed_flow_m3_per_h = max(feed.feed_flow_m3_per_h, 0.001)
    pressure_bar = max(inputs.pressure_bar, 0.1)
    membrane_area = max(inputs.membrane_area_m2, 1.0)

    osmotic_feed = _osmotic_pressure_bar(feed.salinity_g_per_l, feed.temperature_c)
    concentration_factor = 1.0 / max(1.0 - _clamp(inputs.recovery_target, 0.05, 0.85), 0.2)
    osmotic_brine = osmotic_feed * concentration_factor
    avg_osmotic = 0.5 * (osmotic_feed + osmotic_brine)

    net_pressure = pressure_bar - inputs.pretreatment_loss_bar - avg_osmotic
    if net_pressure <= 0:
        warnings.append("Net driving pressure is below zero. Increase pressure or reduce recovery/salinity.")

    lmh = material.water_permeability_a * max(net_pressure, 0.0)
    permeate_flow_from_flux = (lmh * membrane_area) / 1000.0

    permeate_flow_target = feed_flow_m3_per_h * _clamp(inputs.recovery_target, 0.05, 0.85)
    permeate_flow = min(permeate_flow_from_flux, permeate_flow_target)
    brine_flow = max(feed_flow_m3_per_h - permeate_flow, 0.001)

    rejection = _effective_rejection(material, pressure_bar, feed.temperature_c)
    passage = 1.0 - rejection + material.salt_permeability_b * 0.01
    permeate_salinity = max(feed.salinity_g_per_l * passage, 0.02)

    delta_p_pa = pressure_bar * 1e5
    feed_flow_m3_s = feed_flow_m3_per_h / 3600.0
    hydraulic_power_w = delta_p_pa * feed_flow_m3_s
    pump_eff = _clamp(inputs.pump_efficiency, 0.3, 0.95)
    pump_power_kw = hydraulic_power_w / max(pump_eff, 1e-6) / 1000.0

    recovery_device_gain_kw = hydraulic_power_w * _clamp(inputs.energy_recovery_efficiency, 0.0, 0.95) / 1000.0
    net_power_kw = max(pump_power_kw - recovery_device_gain_kw, 0.0)

    if permeate_flow <= 1e-5:
        sec = float("inf")
        warnings.append("No permeate production predicted at current settings.")
    else:
        sec = net_power_kw / permeate_flow

    membrane_stress = (pressure_bar / material.max_pressure_bar) * (avg_osmotic / max(osmotic_feed, 0.1))

    if pressure_bar > material.max_pressure_bar:
        warnings.append("Operating pressure exceeds selected membrane max pressure.")
    if permeate_salinity > 0.6:
        warnings.append("Permeate salinity is high; quality target may not be met.")
    if inputs.recovery_target > 0.6:
        warnings.append("High recovery target can accelerate fouling/scaling.")

    return SimulationResult(
        feed_osmotic_pressure_bar=osmotic_feed,
        net_driving_pressure_bar=net_pressure,
        permeate_flow_m3_per_h=permeate_flow,
        brine_flow_m3_per_h=brine_flow,
        permeate_salinity_g_per_l=permeate_salinity,
        rejection_fraction=rejection,
        specific_energy_kwh_per_m3=sec,
        pump_power_kw=net_power_kw,
        membrane_stress_index=membrane_stress,
        warnings=warnings,
    )
