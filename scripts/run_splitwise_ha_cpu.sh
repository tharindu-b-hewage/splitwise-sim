rq_type=$1
rq_rate=$2
vm_rq_type=$3

# Splitwise-HH
python run.py \
    applications.0.scheduler=mixed_pool \
    cluster=half_half-with-cpu \
    cluster.servers.0.count=0 \
    cluster.servers.1.count=22 \
    cluster.servers.1.sku="$vm_rq_type" \
    start_state=splitwise-with-cpu \
    start_state.prompt.num_instances=5 \
    start_state.token.num_instances=17 \
    performance_model=db \
    trace.filename=rr_"$1"_"$2" \
    seed=0


#python run.py \
#    applications.0.scheduler=mixed_pool \
#    cluster=half_half-with-cpu \
#    cluster.servers.0.count=1 \
#    cluster.servers.1.count=21 \
#    start_state=splitwise-with-cpu \
#    start_state.split_type=heterogeneous \
#    performance_model=db \
#    trace.filename=rr_"$1"_"$2" \
#    seed=0
# 21P and 1T machine count is from power-optimized HA cluster
#    cluster.servers.0.count=26 \
#    cluster.servers.1.count=25 \
