# gyoza
## Generate Your Own kuberneteZ Application-scheduler

Final project for 6.887

### Setup

install `kind` and `kubectl`
https://kind.sigs.k8s.io/docs/user/quick-start/#installation

use `download-spark.sh` to get a local spark installation

`start.sh` and `stop.sh` scripts setup and teardown the kind cluster
Docker must be running on your machine

`kind-cluster-config.yaml` should point to `scheduler-config.yaml`
still TBD if there is an easy way to change scheduler params without restarting the cluster.
Maybe we can use "profiles"