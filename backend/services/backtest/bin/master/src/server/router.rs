use crate::{
    handlers::{
        add_series_handler, add_timeframe_handler, delete_series_handler, edit_series_handler,
        get_state_handler, set_risk_handler, start_backtest_handler, stop_backtest_handler,
    }, master::state::AppState, server::ws::websocket_handler,
};
use axum::{
    Router,
    routing::{get, post},
};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/master/add-timeframe", post(add_timeframe_handler))
        .route("/master/add-series", post(add_series_handler))
        .route("/master/edit-series", post(edit_series_handler))
        .route("/master/delete-series", post(delete_series_handler))
        .route("/master/set-risk", post(set_risk_handler))
        .route("/master/get-state", get(get_state_handler))
        .route("/master/start-backtest", post(start_backtest_handler))
        .route("/master/stop-backtest", post(stop_backtest_handler))
        .route("/master/ws", get(websocket_handler))
}
