# APES-Optimizer

Add link to paper

This repository started as a fork from [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
You can find helpful [discussions](https://github.com/OpenNMT/OpenNMT-py/issues/340) and [explanations](https://github.com/OpenNMT/OpenNMT-py/blob/master/docs/source/Summarization.md) in the repository. 

## Running Models
### Pretraining
Pretrain the model as explained in the likes above. Or download one of the downloadable models in the [this explanations](https://github.com/OpenNMT/OpenNMT-py/blob/master/docs/source/Summarization.md). I used the `ada6_bridge_oldcopy_tagged_acc_54.17_ppl_11.17_e20.pt` model as my pretrained model.

### Fine-tunning
In order to fine tune as noted in the paper

```python train.py -save_model models/entities_attn -data path/to/data/with/filenames -train_from path/to/model/ada6_bridge_oldcopy_tagged_acc_54.17_ppl_11.17_e20.pt -copy_attn -global_attention mlp -word_vec_size 128 -rnn_size 512 -layers 1 -encoder_type brnn -epochs 20 -max_grad_norm 2 -dropout 0. -batch_size 16 -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 -reuse_copy_attn -copy_loss_by_seqlength -bridge -seed 777 -gpuid 0 > entities_attn.txt 2>&1```.

### Generation
Generation is done using the `translate.py` script

```python translate.py -gpu 0 -src_seq_length_trunc 400 -batch_size 20 -beam_size 5 -model path/to/model/ada6_bridge_oldcopy_tagged_acc_54.17_ppl_11.17_e20.pt -src path/to/data/test.txt.src -output testout/file.out -min_length 35 -verbose -stepwise_penalty -coverage_penalty summary -length_penalty wu -alpha 0.9 -beta 0.5 -gamma 0.5 -verbose -block_ngram_repeat 3 -ignore_when_blocking "." "</t>" "<t>" > translating.txt 2>&1 &```

A more up-to-date explanations will be added.