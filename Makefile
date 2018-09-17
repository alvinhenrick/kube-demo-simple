.PHONY: build

build:
	docker build -t alvinhenrick/kube-demo-simple .

login:
	docker login

push:
	docker push alvinhenrick/kube-demo-simple

createpvc:
	kubectl apply -f mystorage.yaml

deletepvc:
	kubectl delete -f mystorage.yaml

copydata:
	kubectl cp ./iris dataaccess:/data/

train:
	kubectl apply -f tfjobsimple.yaml

download:
	kubectl cp dataaccess:/model/iris_model ./iris_model

s2i:
	s2i build . seldonio/seldon-core-s2i-python3:0.1 alvinhenrick/iris-classification:0.1 --env MODEL_NAME=IrisClassifier --env API_TYPE=REST --env SERVICE_TYPE=MODEL --env PERSISTENCE=0

s2ipush:
	docker push alvinhenrick/iris-classification:0.1

serve:
	ks generate seldon-serve-simple iris-classification --image=alvinhenrick/iris-classification:0.1
	ks apply default -c iris-classification

portforward:
	kubectl port-forward `kubectl get pods -n default -l service=ambassador -o jsonpath='{.items[0].metadata.name}'` -n default 8080:80

clean:
	kubectl delete -f tfjobsimple.yaml
	ks delete default -c iris-classification
	ks component rm iris-classification


