# distributed_learning_simulator

This is a simulator for distributed Machine Learning and Federated Learning on a single host. It implements common algorithms as well as our works. The code is currently considered unstable and undergoes modifications over time, so take it with your risk.

## Installation

This is a Python project. The third party dependencies are listed in **requirements.txt**. If you use PIP, it should be easy to install them.

pip3 install -r requirements.txt

## GTG Shapley Value

To run the algorithms of [GTG-Shapley: Efficient and Accurate Participant Contribution Evaluation in Federated Learning](https://dl.acm.org/doi/pdf/10.1145/3501811), use this command

```
source ./env/bin/activate
source /home/matthias089786/venv/bin/activate  

.\env\Scripts\activate
```

```
ulimit -n 4096
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
bash gtg_shapley_train.sh


python ./simulator.py --config-name gtg_sv/mnist.yaml
```

## FedOBD

To run the algorithms of [FedOBD: Opportunistic Block Dropout for Efficiently Training Large-scale Neural Networks through Federated Learning](https://arxiv.org/abs/2208.05174), use this command

```
bash fed_obd_train.sh
```

## Troubleshooting

- https://discuss.pytorch.org/t/libcudnn-cnn-infer-so-8-library-can-not-found/164661
