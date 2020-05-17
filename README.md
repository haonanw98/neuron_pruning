### Neuron Pruning Specification-ver5

Date: May 16th aternoon

To-do: 轮子 of random drop

#### Convenient Usage for Conv4/8

**Pretrain**:

```
python main.py --exp_mode pretraining --config configs/our_yml/conv4_globalchannel.yml --multigpu 0
```

**Pruning**：

```
python main.py --exp_mode pruning --config configs/our_yml/conv4_globalchannel.yml --multigpu 0 --epochs 50 --resume your_pretrain_pth --prune-rate k
```

**Fine-tune**:

```
python main.py --exp_mode finetuning --config configs/our_yml/conv4)globalchannel.yml --multigpu 0 --epochs 50 --lr 0.01 --resume your_pruning_pth --prune-rate k
```



#### Standard Usage

​		Our pruning is a three-step designs. You need to pretrain a network, then prune several neurons, and fine-tune it. During the process, several configs should be identical, i.e. numbers of GPU, prune-rate(which you should only specify in prune process), config_file.

​		**New usage：**

​				you can decide which method to use when ranking scores. Default is absolute, which is identical to  hidden's code, simply  calculating the average of weights. The other is relevant, which calculates the ratio of scores with init score. If relevant type is used, then the second command you  could control is whether_abs, which means：
$$
\begin{align}
\text{relevant,abs} &\quad \text{score$_\text{neuron}$} = \frac{\sum |\text{weight score}|}{\sum |\text{init score}|}\\
\text{relevant,noabs} &\quad \text{score$_\text{neuron}$} = \frac{\sum \text{weight score}}{\sum |\text{init score}|}\\
\end{align}
$$
​		A standard usage is as follows:

**Pretrain** : 

~~~
python main.py --exp_mode pretraining --config config_file.yaml --multigpu gpu_form --name thesamename
~~~

**Prune**:

~~~
python main.py --exp_mode pruning -epochs #epoch(recommend:~50) --config pretraining_cfg_file.yaml --name thesamename --resume pertained_model_dir --pmode yourmode [--rank_method relevant --whether_abs noabs]  --prune-rate k  --multigpu gpu_form 
~~~

**Fine-tune**:

~~~
python main.py --exp_mode finetuning --config pretraining_cfg_file.yaml --epochs #epoch(recommend:~50) --lr 0.01 --name thesamename --resume pruned_model_dir --pmode yourmode [--rank_method relevant --whether_abs noabs] --multigpu gpu_form  --prune-rate k(identical to prune process!!!)
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

​		Four pruning methods are available, including global and layer-wise pruning, and 2 method of each. The default mode is "layer-wise" and "normal" which is exactly the same as hidden code. If you want to change it, consider

```
--pmode normal/filter/channel  --pscale layerwise/global


​		For the global pruning, to avoid the situation that all the channels in a layer will be prune after the initialization, consider warmup

--gp_warm_up --gp_warm_up_epoch 
      
```

random prunning is now available use --prandom to randomly prunning each layer

set fix prune rate for each layer is available under random prunning, please use --prlist to set the prune rate. set prune rate like this:
    --prlist 0.1 0.2 ...  Remember that if the length for the prlist is not enough, other prune rate will be same as prune-rate

protect is also available: --protect "linear"/ "linear\_last". Notice that there are some bugs here(not available for resnet and vgg yet).


