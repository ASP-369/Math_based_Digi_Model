from dataclasses import dataclass


@dataclass(frozen=True)
class MembraneMaterial:
    name: str
    water_permeability_a: float  # LMH/bar
    salt_permeability_b: float  # dimensionless surrogate
    max_pressure_bar: float
    nominal_rejection: float  # fraction 0-1


MEMBRANE_LIBRARY = {
    "Polyamide-Standard": MembraneMaterial(
        name="Polyamide-Standard",
        water_permeability_a=1.6,
        salt_permeability_b=0.10,
        max_pressure_bar=70.0,
        nominal_rejection=0.992,
    ),
    "Polyamide-HighFlux": MembraneMaterial(
        name="Polyamide-HighFlux",
        water_permeability_a=2.0,
        salt_permeability_b=0.14,
        max_pressure_bar=65.0,
        nominal_rejection=0.988,
    ),
    "CelluloseTriacetate": MembraneMaterial(
        name="CelluloseTriacetate",
        water_permeability_a=1.1,
        salt_permeability_b=0.08,
        max_pressure_bar=60.0,
        nominal_rejection=0.985,
    ),
}
