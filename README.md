reference
https://github.com/kentaroy47/vision-transformers-cifar10




show results
python train_cifar100.py --net res50 --n_epochs 0 --lr 0.001 --bs 256 -r 200


resume train    #**  不建議   因為 cosine_scheduler 的 learning rate 已經見底了
python train_cifar100.py --net res50 --n_epochs 3 --lr 0.001 --bs 256 -r 200

train resnet from scratch       #一張3060  大概要跑個幾小時
python train_cifar100.py --net res50 --n_epochs 200 --lr 0.001 --bs 256 --aug    
                                                                        #data aug 可以提昇個1到2%



transfer learning (模型是在google 在  image net 上面訓練過的 vision transformer)
```s
$ python train_cifar100.py --net vit_timm --n_epochs 10   
```
5  epochs test acc 大概  81.5%

timm.list_models('vit*',pretrained=True)   #***可以確認要跑那一個  pretrained 過的 模型   
#我目前是用 vit_tiny_r_s16_p8_224  如果有人顯卡資源夠可以試試看  vit_small 