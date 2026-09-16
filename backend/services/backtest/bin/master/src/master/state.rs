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

use crate::{
    config::AppConfig,
    engine::{
        engine::{EngineState, default_portfolio}
    },
    snapshot::ReplaySnapshot,
    tasks::ReplayStep,
};
use anyhow::Result;
use cursor_db::{binary::BinaryFile, trade::Trade};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::{fmt};
use tokio::sync::{Notify, RwLock, watch};
use tokio::sync::{RwLockReadGuard, watch::Sender};
use tracing::info;
use uuid::Uuid;

//----------------------------------------------------------------------------------------------------------------------
//  MASTER STATE
//----------------------------------------------------------------------------------------------------------------------

pub type Tick = Trade;

#[derive(Debug, Clone, Copy)]
pub struct TickInfo {
    pub tick_index: usize,
    pub tick: Tick,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum ReplayStatus {
    Stopped,
    Running,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MasterStatus {
    Pending,
    Unsync,
    Ready,
}

#[derive(Clone, Serialize)]
pub struct MasterState {
    pub symbol: String,
    pub config_id: String,
    pub status: MasterStatus,
    pub replay_status: ReplayStatus,
    pub replay_step: ReplayStep,
    #[serde(skip)]
    pub tick_data: Arc<BinaryFile>,
    pub tick_index: usize,
    pub engine_state: EngineState,
}

impl fmt::Debug for MasterState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MasterState")
            .field("status", &self.status)
            .field("replay_status", &self.replay_status)
            .field("replay_step", &self.replay_step)
            .field("tick_index", &self.tick_index)
            .field("engine_state", &self.engine_state)
            .finish()
    }
}

impl MasterState {
    #[inline]
    pub fn tick_by_index(&self, index: usize) -> Option<&Tick> {
        self.tick_data.trade(index)
    }

    #[inline]
    pub fn tick_batch(&self, batch_size: usize) -> &[Tick] {
        self.tick_data.range(self.tick_index, batch_size)
    }

    #[inline]
    pub fn current_tick(&self) -> Option<&Tick> {
        self.tick_data.trade(self.tick_index)
    }

    #[inline]
    pub fn has_next_tick(&self) -> bool {
        self.tick_index + 1 < self.tick_data.len()
    }

    pub fn current_tick_info(&self) -> Option<TickInfo> {
        self.current_tick().copied().map(|tick: Tick| TickInfo {
            tick_index: self.tick_index,
            tick,
        })
    }

    pub fn can_publish(&self) -> bool {
        self.status == MasterStatus::Ready && self.replay_status == ReplayStatus::Running
    }
}

//----------------------------------------------------------------------------------------------------------------------
// APP STATE
//----------------------------------------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct AppState {
    pub master: Arc<RwLock<MasterState>>,
    pub boot_id: String,
    pub replay_batch_size: usize,
    pub replay_notify: Arc<Notify>,
    pub engine_notify: Arc<Notify>,
    pub engine_ack_notify: Arc<Notify>,
    pub master_state_tx: Sender<Arc<String>>,
}

impl AppState {
    pub fn new(
        config: AppConfig,
        tick_data: Arc<BinaryFile>,
        snapshot: Option<ReplaySnapshot>,
    ) -> Self {
        info!("Loading snapshot...");

        let boot_id: String = Uuid::now_v7().to_string();

        let replay_batch_size: usize = 300000;

        let (config_id, tick_index, replay_step, engine_state) = match snapshot {
            Some(snapshot) => (
                snapshot.config_id,
                snapshot.tick_index,
                snapshot.replay_step,
                snapshot.engine_state,
            ),

            None => {
                let mut engine_state: EngineState = EngineState::default();
                engine_state.portfolio = default_portfolio(config.initial_cash);

                (
                    Uuid::now_v7().to_string(),
                    0,
                    ReplayStep::PublishTick,
                    engine_state,
                )
            }
        };

        let master_state: MasterState = MasterState {
            symbol: config.symbol,
            config_id,
            status: MasterStatus::Ready,
            replay_status: ReplayStatus::Stopped,
            replay_step,
            tick_data,
            tick_index,
            engine_state
        };

        let (master_state_tx, _) = watch::channel(Arc::new(String::from("{}")));

        Self {
            master: Arc::new(RwLock::new(master_state)),
            boot_id,
            replay_batch_size,
            replay_notify: Arc::new(Notify::new()),
            engine_notify: Arc::new(Notify::new()),
            engine_ack_notify: Arc::new(Notify::new()),
            master_state_tx,
        }
    }

    pub async fn publish_master_state(&self) -> Result<()> {
        let json: String = {
            let master: RwLockReadGuard<'_, MasterState> = self.master.read().await;
            serde_json::to_string_pretty(&*master)?
        };

        let _ = self.master_state_tx.send(Arc::new(json));

        Ok(())
    }
}
