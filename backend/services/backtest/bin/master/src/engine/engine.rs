use pulsar::{DeserializeMessage, Payload};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Series {
    pub id: String,
    pub kind: String,
    pub level: u8,
    pub primary: bool,
    pub overlay: bool,    
    pub params: HashMap<String, Value>,
    #[serde(flatten)]
    pub extra: Option<HashMap<String, Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timeframe {
    pub id: String,
    pub series: HashMap<String, Series>,
    pub timeframe_ms: u64,
    pub live: Option<HashMap<String, Value>>,
    pub closed: Option<HashMap<String, Value>>,
    pub is_new: bool,
    pub is_closed: bool,
}

impl Timeframe {
    pub fn timeframe_ms(id: &str) -> Option<u64> {
        match id {
            "1m" => Some(60_000),
            "5m" => Some(300_000),
            "15m" => Some(900_000),
            "30m" => Some(1_800_000),
            "1h" => Some(3_600_000),
            "4h" => Some(14_400_000),
            "1d" => Some(86_400_000),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EngineStrategy {
    pub kind: String,
    pub params: HashMap<String, Value>,
    #[serde(flatten)]
    pub extra: Option<HashMap<String, Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineState {
    pub tick_index: usize,
    pub time: u64,
    pub timeframes: HashMap<String, Timeframe>,
    pub strategy: EngineStrategy,
    #[serde(default)]
    pub risk: HashMap<String, Value>,
    #[serde(default)]
    pub portfolio: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub orders: Option<HashMap<String, Value>>,
}

impl Default for EngineState {
    fn default() -> Self {
        Self {
            tick_index: 0,
            time: 0,
            timeframes: HashMap::new(),
            strategy: EngineStrategy {
                kind: "Strategy1".to_string(),
                params: HashMap::new(),
                extra: None
            },
            risk: HashMap::new(),
            portfolio: None,
            orders: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStateMessage {
    pub boot_id: String,
    pub config_id: String,
    pub tick_index: usize,
    pub time: u64,
    pub timeframes: HashMap<String, Timeframe>,
    pub strategy: EngineStrategy,
    #[serde(default)]
    pub risk: HashMap<String, Value>,
    #[serde(default)]
    pub portfolio: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub orders: Option<HashMap<String, Value>>,
}

impl DeserializeMessage for EngineStateMessage {
    type Output = Result<EngineStateMessage, rmp_serde::decode::Error>;

    fn deserialize_message(payload: &Payload) -> Self::Output {
        rmp_serde::from_slice(&payload.data)
    }
}
