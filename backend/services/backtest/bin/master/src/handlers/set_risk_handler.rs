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

use crate::master::state::{AppState, MasterState, ReplayStatus};
use axum::{Json, extract::State, http::StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use tokio::sync::RwLockWriteGuard;
use tracing::info;

///
/// Risk configuration schema (forwarded unchanged to the engine):
///
///     {
///         "stop":    { "type": "percent",  "value": 0.01 },
///         "targets": [
///             { "type": "percent", "value": 0.02 },
///             { "type": "percent", "value": 0.03 }
///         ],
///         "trailing": {
///             "enabled": true,
///             "distance": { "type": "percent", "value": 0.005 }
///         }
///     }
///
#[derive(Debug, Deserialize)]
pub struct Request {
    pub config: HashMap<String, Value>,
}

#[derive(Serialize)]
pub struct Response {
    pub success: bool,
    pub message: String,
}

///
/// Sets the risk configuration used by the engine's OrderManager.
///
pub async fn set_risk_handler(
    State(state): State<AppState>,
    Json(req): Json<Request>,
) -> (StatusCode, Json<Response>) {
    let mut master: RwLockWriteGuard<'_, MasterState> = state.master.write().await;

    if master.replay_status != ReplayStatus::Stopped {
        return (
            StatusCode::CONFLICT,
            Json(Response {
                success: false,
                message: "Cannot set risk while replay is running.".to_string(),
            }),
        );
    }

    if master.tick_index != 0 {
        return (
            StatusCode::CONFLICT,
            Json(Response {
                success: false,
                message: "Cannot set risk after the backtest has started.".to_string(),
            }),
        );
    }

    master.engine_state.risk = req.config;

    master.config_id = uuid::Uuid::now_v7().to_string();

    drop(master);

    let _ = state.publish_master_state().await;

    info!("Risk configuration updated.");

    (
        StatusCode::OK,
        Json(Response {
            success: true,
            message: "Risk configuration updated.".to_string(),
        }),
    )
}