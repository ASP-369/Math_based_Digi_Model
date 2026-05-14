# Desalination Mathematical Twin (No Azure Digital Twins)

This project gives you a local, fast, and interactive mathematical twin of an RO desalination plant.
It is designed to be simple, explainable, and easy to connect with your external ML models.

## What You Get

- Component-level mathematical twin:
  - Intake
  - Pretreatment
  - High-pressure pump
  - RO membrane block
  - Energy recovery device
  - Post-treatment output quality
- Interactive UI using Streamlit:
  - Change salinity, temperature, flow, pressure
  - Switch membrane material
  - Tune recovery and efficiencies
  - See energy, quality, flow split, and membrane stress in real time
- ML endpoint hook:
  - Clean REST call function to connect to cloud-hosted models
  - Works with Azure-hosted models via HTTPS endpoint

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run UI:

   ```bash
   streamlit run app.py
   ```

## Workflow (Clear and Practical)

1. Define process inputs in the UI (feed salinity, temperature, flow, pressure).
2. Select membrane material and operating strategy.
3. Run the mathematical twin to get flows, salinity, energy, stress.
4. Send twin input and output features to your ML endpoint.
5. Use ML predictions for:
   - Energy optimization setpoints
   - Membrane fouling or health warning
   - Preventive maintenance decisions
6. Feed optimized setpoints back into twin and validate behavior.

## Can You Use Azure-Hosted ML Models Directly?

Yes. You can call them directly from this local twin UI through REST endpoints.

- Put your endpoint URL and API key in the UI (or environment variables).
- The app sends JSON payload and displays prediction output.
- No dependency on Azure Digital Twins service.

## Project Structure

- app.py: Streamlit interactive UI
- src/twin/materials.py: membrane material library
- src/twin/components.py: typed input/output model containers
- src/twin/plant.py: mathematical process model
- src/twin/ml_client.py: external ML endpoint connector

## Notes on Model Accuracy

This is an explainable first-principles baseline model. For production-grade accuracy:

- Calibrate parameters with historical plant data
- Add multi-stage RO and detailed pressure-drop equations
- Include scaling chemistry and temperature correction factors
- Validate against lab/plant test runs
