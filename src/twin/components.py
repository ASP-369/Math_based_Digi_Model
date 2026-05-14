from dataclasses import dataclass


@dataclass
class IntakeState:
    salinity_g_per_l: float
    temperature_c: float
    feed_flow_m3_per_h: float


@dataclass
class TwinInputs:
    intake: IntakeState
    pressure_bar: float
    pump_efficiency: float
    membrane_area_m2: float
    recovery_target: float
    pretreatment_loss_bar: float
    energy_recovery_efficiency: float


@dataclass
class SimulationResult:
    feed_osmotic_pressure_bar: float
    net_driving_pressure_bar: float
    permeate_flow_m3_per_h: float
    brine_flow_m3_per_h: float
    permeate_salinity_g_per_l: float
    rejection_fraction: float
    specific_energy_kwh_per_m3: float
    pump_power_kw: float
    membrane_stress_index: float
    warnings: list[str]
