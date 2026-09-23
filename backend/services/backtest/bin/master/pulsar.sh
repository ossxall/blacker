#!/bin/bash

ADMIN_URL=http://localhost:16080

docker exec -it backtest_pulsar bin/pulsar-admin --admin-url $ADMIN_URL topics create persistent://public/default/master.tick
docker exec -it backtest_pulsar bin/pulsar-admin --admin-url $ADMIN_URL topics create persistent://public/default/engine.state


docker exec -it backtest_pulsar bin/pulsar-admin --admin-url $ADMIN_URL topics delete persistent://public/default/master.tick
docker exec -it backtest_pulsar bin/pulsar-admin --admin-url $ADMIN_URL topics delete persistent://public/default/engine.state


docker exec -it backtest_pulsar bin/pulsar-admin --admin-url $ADMIN_URL topics unsubscribe \
  --subscription engine-sub \
  persistent://public/default/master.tick