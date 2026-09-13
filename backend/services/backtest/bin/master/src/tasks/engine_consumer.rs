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

use crate::master::state::{AppState, MasterState};
use crate::engine::engine::{EngineState, EngineStateMessage};
use anyhow::{Result, anyhow};
use futures::TryStreamExt;
use pulsar::consumer::Message;
use pulsar::{Consumer, Pulsar, SubType, TokioExecutor};
use std::sync::Arc;
use tokio::sync::RwLockWriteGuard;
use tracing::{error, info};

fn validate_engine_state(
    state: &AppState,
    master: &MasterState,
    engine_state_message: &EngineStateMessage,
) -> Result<()> {
    if engine_state_message.boot_id != state.boot_id {
        return Err(anyhow!("Unexpected boot_id."));
    }

    if engine_state_message.config_id != master.config_id {
        return Err(anyhow!("Unexpected config_id."));
    }

    let remaining: usize = master.tick_data.len() - master.tick_index;

    let expected_last_tick: usize = master.tick_index + remaining.min(state.replay_batch_size) - 1;

    if engine_state_message.tick_index != expected_last_tick {
        return Err(anyhow!(
            "Unexpected engine tick_index. Expected {}, got {}",
            expected_last_tick,
            engine_state_message.tick_index
        ));
    }

    Ok(())
}

pub async fn run(state: AppState, pulsar: Arc<Pulsar<TokioExecutor>>) -> Result<()> {
    // 1. Create Consumer: create pulsar consumer.
    let mut consumer: Consumer<EngineStateMessage, TokioExecutor> = match pulsar
        .consumer()
        .with_topic("persistent://public/default/engine.state")
        .with_subscription_type(SubType::Exclusive)
        .with_subscription("master-sub")
        .build()
        .await
    {
        Ok(con) => con,
        Err(error) => {
            error!(?error, "Failed to create EngineState consumer.");
            return Err(error.into());
        }
    };

    info!("Starting engine consumer...");

    // 1. Start Loop: start the main loop.
    loop {
        //  Receive the next EngineState message from Pulsar.
        let message: Message<EngineStateMessage> = match consumer.try_next().await {
            Ok(Some(msg)) => msg,
            Ok(None) => {
                info!("Engine consumer closed.");
                break;
            }
            Err(error) => {
                error!(?error, "Failed to receive EngineState.");
                continue;
            }
        };

        // Deserialize the received EngineState message.
        let engine_state_message: EngineStateMessage = match message.deserialize() {
            Ok(state) => state,

            Err(error) => {
                error!(?error, "Failed to deserialize EngineState. Aborting.");

                if let Err(error) = consumer.ack(&message).await {
                    error!(?error, "Failed to ACK invalid EngineState.");

                    //TODO: THIS IS FATAL ERROR
                }

                continue;
            }
        };

        // Validate message.
        {
            let mut master: RwLockWriteGuard<'_, MasterState> = state.master.write().await;

            match validate_engine_state(&state, &master, &engine_state_message) {
                Ok(()) => {
                    master.engine_state = EngineState {
                        tick_index: engine_state_message.tick_index,
                        time: engine_state_message.time,
                        timeframes: engine_state_message.timeframes,
                        strategy: engine_state_message.strategy,
                        risk: engine_state_message.risk,
                        portfolio: engine_state_message.portfolio,
                        orders: engine_state_message.orders,
                    };
                }
                Err(reason) => {
                    error!(?reason, "Rejected EngineState ACKing...");

                    drop(master);

                    if let Err(error) = consumer.ack(&message).await {
                        error!(?error, "Failed to early ACK EngineState.");
                    }

                    continue;
                }
            }
        }

        state.engine_notify.notify_one();

        state.engine_ack_notify.notified().await;

        match consumer.ack(&message).await {
            Ok(_) => {
                // info!("EngineState ACK.");
            }
            Err(error) => {
                error!(?error, "Failed to ACK EngineState.");
                continue;
            }
        }
    }

    info!("Engine consumer finished.");

    Ok(())
}
