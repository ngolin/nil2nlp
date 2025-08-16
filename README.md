# `nil2nlp`: New to NLP

本仓库用更简洁和现代的 API 重新实现 [NLP From Scratch](https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) 教程，包括分类器、生成器和翻译器，提供更好的可读性和更好的模型设计。

## 1. 分类器

> https://github.com/lihz6/nil2nlp/blob/master/classifier.ipynb

实现了通过人名猜国别的分类器，所有人名都经过罗马化或抖音化，能够识别 18 个国家的人名。模型的主体是一个 RNN, 能够接收长短不一的人名字符序列输入，并输出一个标签。模型的架构如下：

```
Classifier(
  (rnn): RNN(88, 128)
  (lin): Linear(in_features=128, out_features=18, bias=False)
)
```

## 2. 生成器

> https://github.com/lihz6/nil2nlp/blob/master/generator.ipynb

利用上面分类器的数据还可以实现一个人名生成器，给定一个国家输入即可生成对应国家的人名。模型的主体是一个 LSTM, 能够结合国别的信息以自回归的方式生成人名。模型的架构如下：

```
Generator(
  (lstm): LSTM(106, 128)
  (line): Linear(in_features=128, out_features=106, bias=False)
)
```

## 3. 翻译器

> https://github.com/lihz6/nil2nlp/blob/master/translator.ipynb

实现了法文到英文的翻译，以 seq2seq 模型的 encoder/decoder 为基础，加入 BahdanauAttention 机制实现了简易的法英翻译，但中英翻译的效果不太好。模型架构如下：

```
Translator(
  (encoder): Encoder(
    (emb): Embedding(542, 128)
    (out): Dropout(p=0.1, inplace=False)
    (gru): GRU(128, 128, batch_first=True)
  )
  (decoder): Decoder(
    (emb): Embedding(469, 128)
    (out): Dropout(p=0.1, inplace=False)
    (att): Attention(
      (wa): Linear(in_features=128, out_features=128, bias=True)
      (ua): Linear(in_features=128, out_features=128, bias=True)
      (va): Linear(in_features=128, out_features=1, bias=True)
    )
    (gru): GRU(256, 128, batch_first=True)
    (lin): Linear(in_features=128, out_features=469, bias=False)
  )
)
```
