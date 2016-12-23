# おかえり彼女 (okaeri-kanojo)

## 概要

「ただいま」の音声入力に対し「おかえり」と返答してくれるプログラムです。
話者認識機能で望まれない暴発を防ぎます。

## 環境

* Windows 10
* MinGW32 4.2.0 + MSYS 1.0
* Julius 4.4.1
* Chainer 1.16.0
    * ほか cuDNN, CUDA, Anaconda など必要に応じて

## 使う

以下準備が必要です。

* Julius 最新版のコンパイル + インストール
    * ディクテーションキットも
* Chainer のインストール
* 「おかえり」の音声ファイル

実行については、okaeri ディレクトリにて
```
[Julius のバイナリ] -C [ディクテーションキットのディレクトリ]/main.jconf -C [ディク(ry]/am-gmm.jconf -nolog -quiet -plugindir .
```

と打ち込んでください。あとはマイクに話せば動くはず。

for more information: http://qiita.com/tsubasaogawa/items/3830e4889cccbbb0cffa
