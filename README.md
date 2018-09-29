## How to install Kubeflow

1. Download and unzip Kubeflow to a directory e.g. `~/sdk/kubeflow-0.2.6`

2. Set the environment variable `export KUBEFLOW_REPO=~/sdk/kubeflow-0.2.6`

3. Initialize the `ksonnet` app `ks init simple_demo_ks`

4. Change directory to `cd simple_demo_ks`

5. Install kubeflow core components. 
    
    i. Add kubeflow registry to `ksonnet` app `ks registry add kubeflow "${KUBEFLOW_REPO}/kubeflow"`
    
    ii.Install components 
    ```
    ks pkg install kubeflow/argo
    
    ks pkg install kubeflow/core
    
    ks pkg install kubeflow/examples
    
    ks pkg install kubeflow/katib
    
    ks pkg install kubeflow/mpi-job
    
    ks pkg install kubeflow/pytorch-job
    
    ks pkg install kubeflow/seldon
    
    ks pkg install kubeflow/tf-serving
    ```
6. Generate Kubeflow core component `ks generate kubeflow-core kubeflow-core`

7. Set Kubeflow core parameters 
   
   i. `ks param set kubeflow-core reportUsage false`
   
   ii.`ks param set kubeflow-core usageId $(uuidgen)`

8. Deploy Kubeflow core component to Kubernetes cluster `ks apply default`

9. Generate Seldon core component `ks generate seldon seldon`

10. Set the role `kubectl create clusterrolebinding seldon-admin --clusterrole=cluster-admin --serviceaccount=default:default`

11. Deploy Seldon core component to Kubernetes cluster `ks apply default`


## Train Simple

1. Build docker training image `make build`

2. Login to your personal docker hub account `make login`

3. Push the training image to docker hub repository `make push`

4. Create a Persistent Volume Claim to store data and trained model `make createpvc`

5. Train the model with Kubeflow `make train` reference [TfJob](tfjobsimple.yaml)

## Serve Model

1. Download the model from Persistent Volume Storage to local folder `make download`

2. Wrap the model to docker image for serving via Seldon using [openshift s2i](https://github.com/openshift/source-to-image) `make s2i`

3. Push the serving image to docker hub repository `s2i push`

4. Serve the model with Kubeflow `make serve`

5. Forward the ambassador port to to local host `make portforward`

6. Make predictions from deployed serving image with Kubeflow `make predict`

## Clean up
1. `make clean`

### Tail Log
`kubectl logs -f $(kubectl get pods -l seldon-app=iris-classification -o=jsonpath='{.items[0].metadata.name}') iris-classification`

### Port Forward
`kubectl port-forward $(kubectl get pods -n default -l service=ambassador -o jsonpath='{.items[0].metadata.name}') -n default 8080:80`

## References

[Seldon s2i Wrapper](https://github.com/SeldonIO/seldon-core/blob/master/docs/wrappers/python.md)