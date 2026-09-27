// BLACKER
// Copyright (C) 2026 Juan José Caballero Rey
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation version 3 of the License.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

use crate::engine::engine::EngineStrategy;
use crate::master::state::{AppState, MasterState, ReplayStatus};
use axum::{Json, extract::State, http::StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use tokio::sync::RwLockWriteGuard;
use tracing::info;

///
/// Strategy selection schema (forwarded unchanged to the engine):
///
///     {
///         "kind":   "Strategy1",
///         "params": { "timeframe": "5m", "label_fast": "EMA 20", ... }
///     }
///
#[derive(Debug, Deserialize)]
pub struct Request {
    pub kind: String,
    #[serde(default)]
    pub params: HashMap<String, Value>,
}

#[derive(Serialize)]
pub struct Response {
    pub success: bool,
    message: String,
}

///
/// Selects the strategy the engine evaluates on every tick.
///
/// The engine registers strategies by name, so `kind` must match a key in
/// the Python `STRATEGY_REGISTRY`. Required series differ per strategy:
/// `Strategy1` needs the fast/slow EMA pair on the timeframe named by its
/// `timeframe` param.
///
pub async fn set_strategy_handler(
    State(state): State<AppState>,
    Json(req): Json<Request>,
) -> (StatusCode, Json<Response>) {
    let mut master: RwLockWriteGuard<'_, MasterState> = state.master.write().await;

    if master.replay_status != ReplayStatus::Stopped {
        return (
            StatusCode::CONFLICT,
            Json(Response {
                success: false,
                message: "Cannot set strategy while replay is running.".to_string(),
            }),
        );
    }

    if master.tick_index != 0 {
        return (
            StatusCode::CONFLICT,
            Json(Response {
                success: false,
                message: "Cannot set strategy after the backtest has started.".to_string(),
            }),
        );
    }

    if req.kind.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(Response {
                success: false,
                message: "Strategy kind cannot be empty.".to_string(),
            }),
        );
    }

    let kind = req.kind.trim().to_string();

    master.engine_state.strategy = EngineStrategy {
        kind: kind.clone(),
        params: req.params,
        extra: None,
    };

    master.config_id = uuid::Uuid::now_v7().to_string();

    drop(master);

    let _ = state.publish_master_state().await;

    info!("Strategy set to {kind}.");

    (
        StatusCode::OK,
        Json(Response {
            success: true,
            message: "Strategy updated.".to_string(),
        }),
    )
}
