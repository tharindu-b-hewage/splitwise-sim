rq_type=$1
rq_rate=$2
python run.py \
    applications.0.scheduler=mixed_pool \
    cluster=half_half-with-cpu \
    cluster.servers.0.count=26 \
    cluster.servers.1.count=25 \
    start_state=splitwise-with-cpu \
    start_state.split_type=heterogeneous \
    performance_model=db \
    trace.filename=rr_"$1"_"$2" \
    seed=0
