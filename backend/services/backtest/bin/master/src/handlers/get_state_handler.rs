use crate::master::state::{AppState, MasterState};
use axum::{
    Json,
    extract::{Query, State},
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLockReadGuard;

#[derive(Debug, Serialize)]
pub struct Response {
    pub boot_id: String,
    pub master: MasterState,
}

#[derive(Debug, Default, Deserialize)]
pub struct Params {
    #[serde(default)]
    pub ui: bool,
}

pub async fn get_state_handler(
    State(state): State<AppState>,
    Query(params): Query<Params>,
) -> Json<Response> {
    let master: RwLockReadGuard<'_, MasterState> = state.master.read().await;

    let mut master: MasterState = master.clone();

    // The default payload is the complete state, bars included, because that
    // is what the engine restores from. The aggregated bars carry the
    // per-price footprint (`volume_at_price`), which is unbounded within a
    // bucket and dominates the payload, so the UI asks for a light state with
    // `?ui=true` and gets the bars stripped. They are dropped here instead of
    // on the shared `Timeframe` type: the snapshot still persists them for the
    // engine restore.
    if params.ui {
        for timeframe in master.engine_state.timeframes.values_mut() {
            timeframe.live = None;
            timeframe.closed = None;
        }
    }

    Json(Response {
        boot_id: state.boot_id.clone(),
        master,
    })
}
