# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-09-22

- Added Streamlit web app with modern UI and navigation
- Integrated live weather into predictions
  - OpenWeather (primary) via secrets `OPENWEATHER_API_KEY`
  - WeatherAPI (fallback) via secrets `WEATHERAPI_KEY`
  - City search/autocomplete using OpenWeather Geocoding
  - Latitude/Longitude input and caching with configurable TTL (`WEATHER_CACHE_TTL`)
- Secrets and config documentation
  - `secrets_example.toml`
  - `streamlit_config_example.toml`
  - README updates for secrets and running
- Dataset upgraded to UK locations
  - `generate_dataset.py` now assigns real UK `city`, `region`, `latitude`, `longitude`, and `country`
  - Regenerated `customers.csv` and metadata
- Prediction improvements
  - Use existing customer coordinates to auto-fetch live weather
  - Mini map showing selected customer location
- Analytics enhancements
  - Region and City filters in Dataset Overview and Analytics Dashboard
  - UK Regional Delivery Success map (Mapbox OpenStreetMap)
- UI/UX
  - App title credit: “Created by Braj Patel”
  - Floating top-right clock with timezone selector, dark-mode aware, sidebar toggle
- Robustness fixes
  - Guarded secrets access when secrets.toml is missing
  - Fixed UnboundLocalError for `weather`
  - Fixed indentation issue in Analytics Dashboard

### Breaking changes
- None. New columns were added to `customers.csv`; existing code paths remain compatible.

### Deployment notes
- Ensure secrets are set via Streamlit Secrets or local `.streamlit/secrets.toml`.
- Recommended: install `watchdog` for faster reloads.

