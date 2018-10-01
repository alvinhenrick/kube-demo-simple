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
	cd simple_demo_ks ; ks generate seldon-serve-simple iris-classification --image=alvinhenrick/iris-classification:0.1
	cd simple_demo_ks ; ks apply default -c iris-classification

portforward:
	kubectl port-forward `kubectl get pods -n default -l service=ambassador -o jsonpath='{.items[0].metadata.name}'` -n default 8080:80

predict:
	curl -X POST -H 'Content-Type: application/json' -d '{"data":{"ndarray":[[5.1, 3.3, 1.7, 0.5]]}}' http://localhost:8080/seldon/iris-classification/api/v0.1/predictions

clean:
	kubectl delete -f tfjobsimple.yaml
	cd simple_demo_ks ; ks delete default -c iris-classification
	cd simple_demo_ks ; ks component rm iris-classification


