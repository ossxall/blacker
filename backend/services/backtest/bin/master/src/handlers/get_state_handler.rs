use crate::master::state::{AppState, MasterState};
use axum::{Json, extract::State};
use serde::Serialize;
use tokio::sync::RwLockReadGuard;

#[derive(Debug, Serialize)]
pub struct Response {
    pub boot_id: String,
    pub master: MasterState,
}

pub async fn get_state_handler(State(state): State<AppState>) -> Json<Response> {
    let master: RwLockReadGuard<'_, MasterState> = state.master.read().await;

    let mut master: MasterState = master.clone();

    // The aggregated bars carry the per-price footprint
    // (`volume_at_price`), which is unbounded within a bucket and
    // dominates the payload. Consumers only need the state, so the bars
    // are stripped here instead of on the shared `Timeframe` type: the
    // snapshot still persists them for the engine restore.
    for timeframe in master.engine_state.timeframes.values_mut() {
        timeframe.live = None;
        timeframe.closed = None;
    }

    Json(Response {
        boot_id: state.boot_id.clone(),
        master,
    })
}
