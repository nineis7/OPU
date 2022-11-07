# OPU_Simulator

* fsim directory includes the functional simulator for OPU instruction sequence.

## Docker Build
install docker
```
sh docker_install.sh
```
build
```
sudo docker build -f Dockerfile --tag fsim:1.0 .
```
launch
```
sudo docker run --rm --pid=host\
                     --mount src="$(pwd)",target=/ws,type=bind\
                     -w /ws\
                     -e "CI_BUILD_HOME=/ws"\
                     -e "CI_BUILD_USER=$(id -u -n)"\
                     -e "CI_BUILD_UID=$(id -u)"\
                     -e "CI_BUILD_GROUP=$(id -g -n)"\
                     -e "CI_BUILD_GID=$(id -g)"\
                     -h fsim\
                     --name fsim\
                     -it --net=host\
                     fsim:1.0\
                     /bin/bash
```
detach (can attach later) : ``ctrl+p`` followed by ``ctrl+q``

exit (cannot attach later): ``exit``

## Tiny YOLO test
```
mkdir workspace
cd workspace
pytest -v ../test/utest.py -k test_tiny_yolo
```
