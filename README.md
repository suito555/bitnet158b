論文に書かれていたBitNet1.58bのBitLinear部の実装です。<br>
https://arxiv.org/abs/2310.11453<br>
https://arxiv.org/abs/2402.17764<br>
<br>
論文を読み自力で書いたところもありますが、以下の方々のコードが非常に参考になりました。<br>
https://github.com/Beomi/BitNet-Transformers<br>
https://github.com/frodo821/BitNet-Transformers<br>
https://github.com/kyegomez/BitNet<br>
<br>
参考にさせてもらったコードから大きな変更点としては、activationの順番とdeqauntizeの部分を直しています。<br>
また、Beomiさんのコードのllamaに追加してみたところ、lossは3.9付近でした。(epoch30000ほど)<br>
何か問題があった場合は教えてください。<br>

補足1:torch.compileとの戦い<br>
どうやらtorch.compileはmatmal後に2の累乗以外をかけるとlossがnanになります。おそらく回避できないのと、原因がわかる方がいたら教えてください。<br>
<br>
補足2:モデルサイズと学習率について<br>
モデルサイズを大きくするとともに、論文通りに、あるいはそれよりさらに学習率を下げたほうが良さそうです。(学習率高めだと結構シビア)<br>
