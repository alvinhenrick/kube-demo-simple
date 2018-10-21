## How to install Kubeflow

1. [Download](https://github.com/kubeflow/kubeflow/releases/tag/v0.2.7) and unzip Kubeflow to a directory e.g. `~/sdk/kubeflow-0.2.7`

2. Set the environment variable `export KUBEFLOW_REPO=~/sdk/kubeflow-0.2.7`

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

### Optional Steps. You can use my docker hub image.

1. Build docker training image `make build`

2. Login to your personal docker hub account `make login`

3. Push the training image to docker hub repository `make push`

### <mark>Necessary Steps</mark> to run the training 

1. Create a Persistent Volume Claim to store data and trained model `make createpvc`

2. Copy training data to PVC `make copydata`

3. Train the model with Kubeflow `make train` reference [TfJob](tfjobsimple.yaml)

## Serve Model

### <mark>Optional Steps</mark>. You can use my docker hub image.

1. Download the model from Persistent Volume Storage to local folder `make download`

2. Wrap the model to docker image for serving via Seldon using [openshift s2i](https://github.com/openshift/source-to-image) `make s2i`

3. Push the serving image to docker hub repository `make s2ipush`

### <mark>Necessary Steps</mark> to serve the model. 

1. Serve the model with Kubeflow `make serve`

2. Forward the ambassador port to to local host `make portforward`

3. Make predictions from deployed serving image with Kubeflow `make predict`

## Clean up

1. `make stop`

2. `make clean`

### Tail Log

`kubectl logs -f $(kubectl get pods -l seldon-app=iris-classification -o=jsonpath='{.items[0].metadata.name}') iris-classification`

`kubectl logs -f kube-demo-simple-master-0`

### Port Forward

`kubectl port-forward $(kubectl get pods -n default -l service=ambassador -o jsonpath='{.items[0].metadata.name}') -n default 8080:80`

### Shell Seldon Container

`kubectl exec -it $(kubectl get pods -l seldon-app=iris-classification -o=jsonpath='{.items[0].metadata.name}') --container iris-classification -- /bin/bash`

## References

[Kubernetes](https://kubernetes.io/)

[Kubeflow](https://www.kubeflow.org/)

[Docker](https://www.docker.com/)

[Seldon](https://www.seldon.io/)

[Ksonnet](https://ksonnet.io/)

[Jsonnet](https://jsonnet.org/)

[s2i](https://github.com/openshift/source-to-image)

[Seldon s2i Wrapper](https://github.com/SeldonIO/seldon-core/blob/master/docs/wrappers/python.md)

[Tensorflow](https://www.tensorflow.org/)