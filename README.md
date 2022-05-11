AI 화재 모델은 크게 fire(불꽃)과 smoke(연기) 두가지 모델로 학습 되었습니다.


구체적으로는 fire모델의 경우 주간을 구분하여 학습하였습니다.


모델은 yolor을 기반으로 학습되었으며 각종 작업자 현업을 위하여 학습 후 onnx 컨버팅하여 tracking, 오탐 제거를 위한 후처리 작업을 진행하였습니다.


single gpu train - python3 train.py --batch-size 8 --img 640 640 --data smoke_2c.yaml --cfg cfg/class2.cfg --weights ''  --device 2 --name smoke_2c --hyp hyp.scratch.1280.yaml --epochs 300

multi gpu train - python3 -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py --batch-size 24 --img 640 640 --data sp_smoke.yaml --cfg cfg/person.cfg --weights '' --device 0,1,2 --sync-bn --name sp_smoke—hyp hyp.scratch.1280.yaml --epochs 300


