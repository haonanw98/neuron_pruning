### Neuron Pruning Specification-ver4

Date: May 11th aternoon

To-do: adding global pruning method.

#### Standard Usage

​		Our pruning is a three-step designs. You need to pretrain a network, then prune several neurons, and fine-tune it. During the process, several configs should be identical, i.e. numbers of GPU, prune-rate(which you should only specify in prune process), config_file.

​		A standard usage is as follows:

**Pretrain** : 

~~~
python main.py --exp_mode pretraining --config config_file.yaml --multigpu gpu_form --name thesamename
~~~

**Prune**:

~~~
python main.py --exp_mode pruning --config pretraining_cfg_file.yaml --resume pertained_model_dir --prune-rate k --epochs #epoch(recommend:~20) --multigpu gpu_form --name thesamename
~~~

**Fine-tune**:

~~~
python main.py --exp_mode finetuning --config pretraining_cfg_file.yaml --resume pruned_model_dir --epochs #epoch(recommend:~20) --lr 0.01 --multigpu gpu_form --name thesamename --prune-rate k(identical to prune process!!!)
~~~

​		Remember, the config "name" should be the same during the whole process, and the program will automatically add prefix to help you specify step.

​		Besides, remember to use "--data" to locate your datasets.

#### Other Mode

**Shuffle**:

​		If you want to examine network's expressivity, do remember to add

```
--shuffle 1 --seed yourseed(deafult is none)
```

​		during the whole process. 

**Prune Method**:

​		Six pruning methods are available, including global and layer-wise pruning, and 3 method of each. The default mode is "layer-wise" and "normal" which is exactly the same as hidden code. If you want to change it, consider

```
--pmode normal/filter/channel  --pscale layerwise/global(this is not supported yet)
```

