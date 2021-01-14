- [Simple Chit-Chat based on KoBART](#simple-chit-chat-based-on-kobart)
  - [Purpose](#purpose)
  - [Install](#install)
  - [How to Train](#how-to-train)

# Simple Chit-Chat based on KoBART 


## Purpose

- [공개된 한글 챗봇 데이터](https://github.com/songys/Chatbot_data)와 pre-trained [KoBART](https://github.com/SKT-AI/KoBART)를 이용한 간단한 챗봇 실험
- `KoBART`의 다양한 활용 가능성을 타진하고 성능을 정성적으로 평가한다.
- fork한 repo를 제 입맛에 맞춰 수정하였습니다.

## Install

```python
git clone https://github.com/L0Z1K/KoBART-chatbot.git
pip install -r requirements.txt
```

## How to Train

1. Fine-Tuning & Chat

```bash
$ CUDA_VISIBLE_DEVICES=0 python script/train.py --max_epochs 3 --gpus 1 --train --chat
```

2. Only Chat with latest saved model

```bash
$ python script/train.py --chat
```
