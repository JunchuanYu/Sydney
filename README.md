![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/SYDNEYAI.jpg)
#    Sydney-AI，一个免费的ChatGPT 平台
我们基于最新的 OpenAI 接口 ChatGPT-3.5-turbo构建了一个ChatGPT web 平台，为「45 度科研人」的朋友们提供免费的服务，经过 2 周的测试平台运行稳定，如果大家在使用中遇到问题也请告诉我们，平台运营的费用由「45 度科研人」公众号承担，也感谢大家的支持！
访问地址: https://junchuanyu-sydney-ai.hf.space

![](https://dunazo.oss-cn-beijing.aliyuncs.com/blog/6.JPG)
## Sydney-AI 对科研人有哪些帮助
 相信大家对 ChatGPT 已经都有所了解，我们认为对于科研人员来说 ChatGPT 是一款很好的辅助软件，在资料搜集，语言润色，辅助开发，文字编辑等方面都有令人印象深刻的表现，在与 Sydney-AI 深度接触的 2 周时间里，团队小伙伴们已经习惯了与它一共配合完成工作。如果你想更深入了解 ChatGPT 或是New Bing 请看之前的一篇文章 。
## Sydney-AI有什么特点？为什么叫这个名字
 能让大家方便的使用 ChatGPT 是我们开发这个应用的初衷，Sydney-AI 特点在于解决国内使用 ChatGPT 的两个痛点，即网络和跨设备问题。
- Sydney-AI 无需魔法即可访问
- Sydeny-AI 在电脑端和手机端都可以使用
- Sydney 这个名字是新版 ChatGPT 对自己的称呼，在 New Bing 的初测阶段 ChatGPT 展现出非常丰富的情感，会幽默，会顶嘴，甚至 会PUA ，很多人都十分喜爱这个表现的不那么像个机器人的 Sydney。然而，由于种种原因，微软对New Bing 进行了更新而情感丰富的 Sydney 消失了。因而取名叫Sydney-AI 是表达对新开放 API 接口的一种期待。
## 如何使用Sydney-AI？
### 软件的基本设定：
- 对话次数没有限制，但限制了每次对话的 token 数为 3000，超过该数值需要重新开启一个对话，token 不同于单词数，1000 个 token 大概相当于 750 个字。
- Sydney-AI 是免费使用的，已经内置了 OpenAI 的 API Key，如果你想要突破 3000 个 token 的限制，也可以在 API Key 的窗口中填写上自己的 API Key。
 - 软件中内设了一些 ChatGPT 的常用角色，可以通过下拉菜单选取，默认的是原始的 ChatGPT。你也可以在输入文本中加入一些角色描述让 ChatGPT 扮演你想要的角色，比如“你是一个遥感专业大学老师，请以简单易懂的方式向小学生科普一下遥感的概念”，再比如“我希望你能担任英语翻译、拼写纠正和改进的角色。我会用任何语言与你交流，你将检测语言，将其翻译并用更正和改进后的英文回答"
### 使用方法
- 在文本框中输入你的问题即可
- 提供了两个可调整的参数，当 Max tokens 数设定为大于 3000 需要输入自己的 OpenAI Key，实测 3000 已能满足大部分使用场景；当 Temperature设定的数值越大，回答越发散，但出错的概率也越大。
- OpenAI API调用方法，当你需要 ChatGPT 完成更有难度的工作，比如将回复文本转换成 csv 文件并下载到本地，这时你需要根据需求对 API 进行功能拓展。下面提供了新版 API 的调用方式，建议使用Colab 进行调试。
```python
import openai
# enter your openai api key
openai.api_key = ”OPENAI_API_KEY“
# enter your prompt
prompt = ‘enter centence here’
response = openai.ChatCompletion.create(
    model=‘gpt-3.5-turbo ’,
    messages=[
        {”role“: ”system“, ”content“: ”you are a helpful assistant.“},
        {”role“: ”user“, ”content“: prompt}
    ]
)
# get the response
resText = response.choices[0].message.content
print(resText)
```


----------------------------------------
请关注微信公众号【45度科研人】获取更多精彩内容，欢迎后台留言！

<div align=center><img width = '300' height ='300' src ="https://dunazo.oss-cn-beijing.aliyuncs.com/blog/wechat-simple.png"/></div>
