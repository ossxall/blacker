use anyhow::{Context, Result};
use std::env;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub symbol: String,
    pub pulsar_url: String,
    pub tick_data_path: String,
    pub initial_cash: f64,
}

impl AppConfig {
    pub fn from_env() -> Result<Self> {
        let symbol: String =
            env::var("SYMBOL").context("Missing required environment variable: SYMBOL")?;

        let pulsar_url: String =
            env::var("PULSAR_URL").unwrap_or_else(|_| "pulsar://localhost:6650".to_string());

        let tick_data_path: String = env::var("TICK_DATA_PATH")
            .context("Missing required environment variable: TICK_DATA_PATH")?;

        let initial_cash: f64 = env::var("INITIAL_CASH")
            .context("Missing required environment variable: INITIAL_CASH")?
            .parse()
            .context("INITIAL_CASH must be a valid float")?;

        Ok(Self {
            symbol,
            pulsar_url,
            tick_data_path,
            initial_cash,
        })
    }
}
