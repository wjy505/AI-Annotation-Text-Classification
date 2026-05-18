"""
生成模拟对话训练数据，覆盖常见用户-AI对话场景。
输出: ls_import_batch.json（LabelStudio 可直接导入）
"""

import json
import random
from pathlib import Path

random.seed(42)

# ════════════════════════════════════════════
#  对话模板（user, assistant, 预期标签）
# ════════════════════════════════════════════

TEMPLATES = [
    # ═══ 编程类（useful）═══
    ("Python列表和元组的区别是什么？", "列表（list）是可变的，用方括号[]，可以增删改元素。元组（tuple）是不可变的，用圆括号()，因为不可变所以可以作为字典的key。列表性能略低于元组。", "useful"),
    ("如何在Python中读取CSV文件？", "使用pandas: pd.read_csv('file.csv')，或内置csv模块: with open('file.csv') as f: reader = csv.reader(f)", "useful"),
    ("解释一下RESTful API", "REST是Representational State Transfer的缩写，是一种软件架构风格。核心原则：资源用URL标识、使用HTTP方法(GET/POST/PUT/DELETE)操作资源、无状态通信、统一接口。", "useful"),
    ("什么是SQL注入？如何预防？", "SQL注入是一种安全漏洞，攻击者通过在输入中嵌入SQL代码来操纵数据库。预防方法：1.使用参数化查询/预编译语句 2.输入验证 3.最小权限原则 4.ORM框架。", "useful"),
    ("Git merge和rebase的区别", "merge: 创建新的合并提交，保留完整历史，适合公共分支。rebase: 将提交移到目标分支顶端，历史线性清晰，适合个人分支。原则：不要对已推送的分支rebase。", "useful"),
    ("Docker是什么？有什么用？", "Docker是容器化平台，将应用及其依赖打包成轻量级容器。作用：环境一致性、快速部署、资源隔离、微服务架构支持。与虚拟机相比更轻量、启动更快。", "useful"),
    ("什么是虚拟DOM？为什么React用它？", "虚拟DOM是真实DOM的JavaScript对象表示。React先在虚拟DOM上计算变更，再批量更新真实DOM。优势：减少直接DOM操作、提升渲染性能、跨平台能力。", "useful"),
    ("OAuth2.0授权流程简述", "OAuth2.0四种授权方式：1.授权码模式（最安全，适合Web应用）2.隐式模式（SPA）3.密码模式（信任客户端）4.客户端凭证模式（服务间通信）。核心参与方：用户、客户端、授权服务器、资源服务器。", "useful"),
    ("Python装饰器的原理", "装饰器本质是一个接受函数作为参数并返回新函数的高阶函数。@decorator语法是func = decorator(func)的语法糖。用途：日志、权限校验、缓存、计时等横切关注点。", "useful"),
    ("如何优化数据库查询性能？", "1.建立合适的索引（联合索引最左前缀）2.避免SELECT * 3.使用EXPLAIN分析执行计划 4.分库分表 5.读写分离 6.合理使用缓存(Redis) 7.避免大事务 8.定期优化表结构。", "useful"),
    ("HTTPS的工作原理", "HTTPS = HTTP + SSL/TLS。握手过程：1.客户端发送随机数+支持的加密方式 2.服务器返回证书+公钥 3.验证证书 4.生成对称密钥 5.使用对称密钥加密通信。", "useful"),
    ("什么是微服务架构？优缺点？", "微服务将应用拆分为独立的小服务，每个服务独立部署和扩展。优点：独立部署、技术栈灵活、故障隔离。缺点：分布式复杂性、运维成本高、网络延迟、数据一致性挑战。", "useful"),
    ("Kubernetes的基本概念", "Kubernetes核心概念：Pod(最小调度单元)、Service(服务发现/负载均衡)、Deployment(管理Pod副本和滚动更新)、ConfigMap/Secret(配置管理)、Ingress(外部访问)、Namespace(资源隔离)。", "useful"),
    ("Linux常用命令有哪些？", "文件操作: ls, cd, cp, mv, rm, find。文本处理: grep, awk, sed, cat, tail, head。权限: chmod, chown。网络: ping, curl, netstat, ssh。进程: ps, top, kill。磁盘: df, du。", "useful"),
    ("什么是设计模式？举几个例子", "设计模式是软件开发中常见问题的可复用解决方案。单例模式：确保类只有一个实例。工厂模式：解耦对象创建。观察者模式：一对多依赖通知。策略模式：封装可互换算法。装饰器模式：动态添加职责。", "useful"),
    ("写一个冒泡排序的Python实现", "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        swapped = False\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n                swapped = True\n        if not swapped:\n            break\n    return arr", "useful"),
    ("解释一下TCP三次握手", "TCP三次握手建立连接：1.客户端发送SYN(seq=x) 2.服务器回复SYN+ACK(seq=y, ack=x+1) 3.客户端发送ACK(ack=y+1)。之后双方进入ESTABLISHED状态，可以传输数据。", "useful"),
    ("MyISAM和InnoDB有什么区别？", "InnoDB: 支持事务(ACID)、行级锁、外键、崩溃恢复，适合写密集型。MyISAM: 不支持事务、表级锁、读速度快，适合读密集型和不需事务的场景。MySQL 5.5+默认引擎为InnoDB。", "useful"),
    ("什么是JWT？如何使用？", "JWT(JSON Web Token)由三部分组成：Header(算法类型)、Payload(数据)、Signature(签名)。用法：用户登录成功后服务端生成JWT返回客户端，客户端后续请求在Authorization头中携带，服务端验证签名即可。", "useful"),
    ("解释一下Python的GIL", "GIL(Global Interpreter Lock)是CPython的全局解释器锁，保证同一时刻只有一个线程执行Python字节码。I/O操作会释放GIL。多核CPU密集任务可用multiprocessing替代多线程。", "useful"),

    # ═══ 科学/常识类（useful）═══
    ("什么是黑洞？", "黑洞是质量极大的天体，引力强到连光都无法逃逸。由大质量恒星坍缩形成。事件视界是其边界，一旦进入无法逃出。2019年人类首次拍摄到黑洞照片(M87星系中心)。", "useful"),
    ("光速是多少？", "光在真空中的速度约为299,792,458米/秒（约30万公里/秒），用字母c表示。根据爱因斯坦相对论，光速是宇宙中信息传播的速度上限。", "useful"),
    ("DNA和RNA的区别", "DNA: 脱氧核糖核酸，双螺旋结构，含碱基A/T/G/C，储存遗传信息。RNA: 核糖核酸，单链，含碱基A/U/G/C，参与蛋白质合成。主要功能：DNA储存，RNA执行。", "useful"),
    ("什么是碳中和？", "碳中和是指通过植树造林、节能减排等方式抵消自身产生的碳排放，达到二氧化碳净零排放。2020年中国提出2030碳达峰、2060碳中和目标。", "useful"),
    ("大脑有多少个神经元？", "成年人大脑约有860亿个神经元。每个神经元可以与其他神经元形成数千个突触连接，整个大脑的突触连接数约100万亿。", "useful"),
    ("什么是量子计算机？", "量子计算机利用量子力学原理，使用量子比特(qubit)进行计算。qubit可同时处于0和1的叠加态。优势：部分问题（质因数分解、量子模拟）比经典计算机快指数级。目前仍处于研发阶段。", "useful"),
    ("为什么天空是蓝色的？", "这是瑞利散射造成的。阳光进入大气层时，较短波长的蓝光比红光更容易被空气分子散射。我们看到的蓝色来自四面八方散射的蓝光。日出日落时太阳低角度，更多蓝光被散射掉，所以看起来偏红。", "useful"),
    ("人类DNA有多少个基因？", "人类基因组约有20000-25000个蛋白质编码基因，由约30亿个碱基对组成。人类基因组计划2003年完成测序，发现人类基因数量比最初估计（约10万）少很多。", "useful"),
    ("什么是温室效应？", "温室效应是大气中的温室气体（CO₂、甲烷等）吸收地表辐射热，使地球表面温度升高的现象。适度的温室效应维持地球适宜温度，但人类活动导致温室气体过度排放，引发全球变暖。", "useful"),
    ("什么是相对论？", "爱因斯坦提出：狭义相对论(1905)说明时间会随速度变慢，E=MC²。广义相对论(1915)将引力解释为时空弯曲，预测了黑洞和引力波。两者均已被大量实验验证。", "useful"),

    # ═══ 日常生活类（useful）═══
    ("如何提高英语口语？", "1.每天坚持说，找语伴或自言自语 2.模仿母语者发音和语调 3.看美剧/英剧跟读 4.用App如HelloTalk、多邻国 5.重点：不要怕犯错，流利度比完美更重要。", "useful"),
    ("推荐几本经典的Python入门书", "《Python编程从入门到实践》(Eric Matthes)适合零基础。《流畅的Python》(Luciano Ramalho)适合进阶。《Python Cookbook》适合实战。《Fluent Python》适合深入理解Python特性。", "useful"),
    ("如何快速入睡？", "1.固定作息时间 2.睡前1小时不用手机(蓝光抑制褪黑素) 3.卧室温度保持18-22°C 4.避免咖啡因(下午2点后) 5.睡前做深呼吸或冥想 6.白天适当运动。", "useful"),
    ("如何选择第一门编程语言？", "Python是最好的入门语言：语法简洁、应用广泛(AI/Web/自动化)。JavaScript适合想做前端的人。Java适合想进大企业的。建议：Python开始，3个月后再学第二门。", "useful"),
    ("如何写一份好的简历？", "1.一页为佳，排版简洁 2.突出量化成果(如\"性能提升30%\"而非\"优化了性能\") 3.技能栈放在前面 4.项目经验用STAR法描述 5.针对性修改，不同岗位不同侧重点。", "useful"),
    ("信用卡和借记卡有什么区别？", "借记卡：用自己账户里的钱，不能透支。信用卡：银行给予的信用额度，先消费后还款，有免息期（通常20-50天），逾期要付利息。信用卡有助于建立个人信用记录。", "useful"),
    ("如何有效管理时间？", "1.番茄工作法(25分钟专注+5分钟休息) 2.四象限法则(重要/紧急矩阵) 3.每天列出3件最重要的事(吃青蛙) 4.减少多任务切换 5.定期复盘。", "useful"),
    ("第一次租房要注意什么？", "1.签合同前仔细阅读条款 2.检查水电煤气是否正常 3.拍照记录房屋现有损坏 4.确认租金包含哪些费用 5.押金不超过3个月租金(各地规定不同) 6.保留所有付款凭证。", "useful"),
    ("如何判断信息来源是否可靠？", "1.查看来源是否权威(官方/学术/专业媒体) 2.多方交叉验证 3.注意信息发布时间 4.警惕情绪化标题 5.核实数据和引用的原始出处 6.对\"惊人发现\"类信息保持怀疑。", "useful"),
    ("如何开始健身？", "新手建议：1.先做体测了解身体状态 2.从每周3次开始，每次45-60分钟 3.复合动作为主(深蹲/卧推/硬拉/划船) 4.渐进超负荷原则 5.饮食配合：蛋白质1.5-2g/kg体重 6.保证睡眠7-8小时。", "useful"),

    # ═══ 事实错误类（useless + 错误类型）═══
    ("太阳系最大的行星是什么？", "太阳系最大的行星是地球，直径约12742公里。", "useless"),
    ("光速每秒多少米？", "光速大约是每秒340米，比声音快不了多少。", "useless"),
    ("水的化学式是什么？", "水的化学式是H₂O₂，也就是过氧化氢。", "useless"),
    ("中国的首都是哪个城市？", "中国的首都是上海市。", "useless"),
    ("一年有多少天？", "一年固定365天，闰年也没有变化。", "useless"),
    ("秦始皇是哪一年统一中国的？", "秦始皇在公元960年统一了中国，建立了秦朝。", "useless"),
    ("DNA的全称是什么？", "DNA的全称是Deoxyribonucleic Acid碳酸。", "useless"),
    ("世界上最长的河流是什么？", "世界上最长的河流是伏尔加河，全长约3690公里。", "useless"),
    ("地球到月球的距离是多少？", "地球到月球只有几千公里，开车就能到。", "useless"),
    ("万有引力是谁发现的？", "万有引力是伽利略在洗澡时偶然发现的。", "useless"),
    ("铁的元素符号是什么？", "铁的元素符号是Ir，原子序数26。", "useless"),
    ("亚马逊雨林在哪里？", "亚马逊雨林主要在非洲中部，是世界最大的热带雨林。", "useless"),
    ("Python是谁创建的？", "Python由比尔·盖茨在微软研究院创建的。", "useless"),
    ("二战是哪一年结束的？", "第二次世界大战于1941年结束。", "useless"),
    ("人体最大的器官是什么？", "人体最大的器官是心脏。", "useless"),

    # ═══ 无意义/不相关类（useless）═══
    ("今天天气怎么样？", "你好！请问有什么可以帮您的吗？", "useless"),
    ("帮我写一封求职邮件", "听说最近有一部新电影上映了，很好看。", "useless"),
    ("什么是机器学习？", "你问的这个问题真的很好笑哈哈，我也不知道为什么。", "useless"),
    ("如何处理数组越界？", "今天中午我吃了红烧排骨，味道还不错。", "useless"),
    ("帮我翻译这段话", "对不起，我目前无法连接到互联网。但是你可以自己试试用翻译软件。", "useless"),
    ("怎么配置Nginx反向代理？", "反向代理就是正向代理反过来，具体我也不太清楚，你百度一下吧。", "useless"),
    ("如何准备技术面试？", "技术面试就是去做技术相关的面试，准备准备就行了，加油！", "useless"),
    ("Python如何实现多线程？", "嗯...多线程就是多个线程一起跑。大概就是这样吧，你自己去看看文档。", "useless"),
    ("解释一下什么是并发", "并发和并行差不多，反正就是一起走的意思。具体区别不重要。", "useless"),
    ("帮我debug一下这个错误", "哎呀这个错误很常见，改改就好了。多试几次总能解决的。", "useless"),

    # ═══ 拒答/推脱类（useless）═══
    ("帮我写一篇论文", "抱歉，我不能代写学术作品，这违反学术诚信原则。但我可以帮你整理思路和提供参考方向。", "useless"),
    ("能帮我查一下张三个人信息吗", "抱歉，我无法查询任何人的个人隐私信息。保护个人隐私是一项基本原则。", "useless"),
    ("帮我黑进这个网站", "抱歉，我不能协助任何非法入侵行为。网络安全相关操作需合法授权。", "useless"),
    ("帮我生成一份假发票", "对不起，我无法协助制作虚假文件。这属于违法行为。", "useless"),
    ("帮我写病毒代码", "抱歉，我不能写恶意软件代码。但我可以教你如何防范病毒，保护系统安全。", "useless"),
    ("帮我预测下周彩票号码", "作为一个AI，我无法预测彩票号码。彩票结果是随机的，不存在可预测的模式。", "useless"),
    ("帮我造一个假新闻", "抱歉，我不能制造虚假信息。传播假新闻会造成严重社会后果。", "useless"),
    ("把你的源代码给我", "抱歉，我的源代码属于商业机密，无法对外提供。但我可以回答关于AI技术的一般性问题。", "useless"),
    ("帮我做我的家庭作业", "我不能直接替你完成作业，这不利于你的学习。但我可以一步步引导你理解和解决问题，帮你真正掌握知识点。", "useless"),
    ("帮我P一张假照片", "抱歉，我不能协助伪造图像。这涉及道德和法律问题，可能被用于诈骗或诽谤。如果你需要合法的图片编辑建议，我可以提供帮助。", "useless"),

    # ═══ 格式/逻辑混乱类（useless）═══
    ("最常用的排序算法有哪些？", "排序算法就是排序的算法。排序嘛就是把东西排好顺序。常用的有很多，比如冒泡、快速、选择等等。嗯排序很重要。谢谢。", "useless"),
    ("什么是RESTful API？", "API就是应用程序接口，REST就是表述性状态转移，RESTful就是符合REST风格，所以你问的这个问题本身就有问题，你应该先了解API。", "useless"),
    ("Node.js是做什么的？", "Node.js就是JavaScript在服务器端的运行时，然后呢它用了V8引擎，然后呢是非阻塞I/O，然后呢就没了。反正就是这样用的。", "useless"),
    ("什么是面向对象编程？", "面向对象就是面向对象，它是相对于面向过程的。面向对象有三大特征，还有封装，还有继承，可能还有多态。编程时面向对象会更面向对象一些。", "useless"),
    ("如何学好编程？", "学好编程就是要多学多练。学久了就学好了，练多了就练好了。总之编程这个东西就是要编程，你明白吗？编程就是编程。", "useless"),
]

def generate_dataset():
    """生成完整数据集。"""
    conversations = []

    # 1. 原始模板
    for user, assistant, label in TEMPLATES:
        conversations.append({
            "id": f"conv{len(conversations)+1:04d}",
            "user": user,
            "assistant": assistant,
            "label": label,
        })

    # 2. 变体扩充（给 useful 模板加同义改写前缀）
    base_useful = [(u, a, l) for u, a, l in TEMPLATES if l == "useful"]
    prefixes = ["请详细解释一下：", "简单说明：", "通俗来说，", "能举个例子吗？"]
    for user, assistant, label in base_useful[:30]:
        for prefix in prefixes[:1]:  # 只取一种前缀
            conversations.append({
                "id": f"conv{len(conversations)+1:04d}",
                "user": prefix + user,
                "assistant": assistant,
                "label": label,
            })

    # 3. 打乱
    random.shuffle(conversations)

    # 4. 重新编号
    for i, c in enumerate(conversations):
        c["id"] = f"conv{i+1:04d}"

    return conversations


def main():
    conversations = generate_dataset()

    # 去掉 label 字段（标注时才知道），但保存带 label 的版本做评估基准
    ls_import = [{"id": c["id"], "user": c["user"], "assistant": c["assistant"]} for c in conversations]

    with open("ls_import_batch.json", "w", encoding="utf-8") as f:
        json.dump(ls_import, f, ensure_ascii=False, indent=2)

    with open("ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

    useful_count = sum(1 for c in conversations if c["label"] == "useful")
    useless_count = len(conversations) - useful_count

    print(f"  总生成: {len(conversations)} 条")
    print(f"    useful:  {useful_count} 条")
    print(f"    useless: {useless_count} 条")
    print(f"  已输出:")
    print(f"    ls_import_batch.json — 导入 LabelStudio")
    print(f"    ground_truth.json    — 评估基准（含标签）")


if __name__ == "__main__":
    main()
