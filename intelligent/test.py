import inspect
from string import punctuation
from string import digits
import re

filter_words = '转发微博'
rule1 = re.compile(u'[^a-zA-Z.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'+digits+punctuation+'\u4e00-\u9fa5]+')
rule2 = re.compile(u'^(转发微博|分享图片){1,15}|(\[超话\]|http:){1,15}|(\.\.\. 全文$){1}')
rule3 = re.compile(u'^回复@.*:')
rule4 = re.compile(u'(?P<pound_left>[\s\b，]*?#)(?P<content>.*?)(?P<pound_right>#[\s\b，]*?)')  # *? 表示懒惰搜索,匹配最短的  #熊猫守护者# --> 熊猫守护者 (有些问题)
rule5 = re.compile(u'(?P<bracket_left>\[)(?P<content>.*?)(?P<bracket_right>\])')  #
# 俊杰 万分的舍不得，希望还有机会再见面 嘉伦的刘海的微博视频
# 用('[\s\b，].*?的微博视频$' ) 进行匹配, 会从俊杰后面的空格处开始匹配;
# 原因: 比懒惰／贪婪规则的优先级更高：最先开始的匹配拥有最高的优先权
#字符串反转处理

rule6 = re.compile(u'^[\s\b，]*?(频视博微的|频视拍秒的).*?[\s\b，]+?')#希望还有机会再见面 嘉伦的刘海的微博视频
# rule7 = re.compile(u'全文$')  #
# rule8 = re.compile(u'\b(\w+)\b\s+\1\b')  #  怎么,过滤重复的字
# ,(\w+),\1\b   是因为我们梁山没售后,允悲,允悲,允悲  --.  允悲
# 将匹配的数字乘以 2

def replaced_after(matched):
    # print("function1->"+matched.group('content'))

    return  '，' + matched.group('content') + '，'

def waste_match_last(matched):
    # print("function1->" + matched.group('back_sentence'))
    if re.search(rule6,matched.group('back_sentence')):
        return matched.group()+waste_match_last(re.search(rule6,matched.group('back_sentence')))
    else:
        return matched.group()

def clean_sentence(line1):
    # TODO 1.√ 两个#中间是话题; 可以存起来; 然后过滤掉2. @后面是人名,用他来代替? ;  3.√ 很多引用的,去重; 4. 去掉http链接
    # √ TODO 5.√ //之前是引用, 从前往后检索, 6.√ 转发微博(过滤)  6.√ [表情]  表情是很重员的情感分析的部分,怎么处理; 为表情创建一个字典?还是只去掉中括号(决定用逗号代替括号)
    # √TODO 6. √嘉伦的刘海的微博视频 7. (不处理)啊啊啊啊啊啊啊啊 8. ヽ(‘⌒´メ)ノ 8 . （分享自 @音悦台 ）
    # TODO 9 √查看图片  10  回复@XXX:  11.  ... 全文 12.微博饭票 13 位置: 句子最后, 城市·地点
    # TODO 13 做一个统计, 空格之间常见短语过滤  14. 《xxx》书名 15. 过滤英文
    print('--------------------------')
    print("rule0->" + line1)
    contents = line1.split('//', 1)

    if contents[0]:
        line1 = re.sub(rule1, '', contents[0])
        if line1.isspace():
            return ''
        else:
            print("rule1->" + line1)
            line2 = re.sub(rule2, '', line1)
        if line2.isspace():
            return ''
        else:
            print("rule2->" + line2)
            line3 = re.sub(rule3, '', line2)
        if line3.isspace():
            return ''
        else:
            print("rule3->" + line3)
            line4 = re.sub(rule4, replaced_after, line3)
            print("rule4->" + line4)
        if line4.isspace():
            return ''
        else:
            line5 = re.sub(rule5, replaced_after, line4)
        if line5.isspace():
            return ''
        else:
            print("rule5->" + line5)
            line5 = line5[::-1]
            # print("reverse rule5->" + line5)
            line6 = re.sub(rule6, '', line5)
        # print("rule6->" + line6)
        result = line6[::-1]
        # line8 = re.sub(rule8, '，', line4)
        # print("rule8->" + line8)
        print("result->" + result)
        return result
    else:
        return ''



def read_data(flag):
    temp_file = './data/temp.txt'
    print('hello')
    if flag:
        result = []
        with open(temp_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                new_line = clean_sentence(line)
                if not new_line.isspace() and len(new_line) >= 1:
                    if new_line.endswith('\n'):
                        result.append(new_line)
                    else:
                        result.append(new_line+'\n')
        with open('./data/temp_result.txt', 'w',encoding='utf-8') as f:
            for line in result:
                f.write(line)
    if not flag:
        line = '#熊猫守护者# #保卫者# No.234243快来'
        line = '是因为我们梁山没售后[允悲][允悲][允悲] '
        line = '#日行一善[超话]# 人民有信仰'
        line = '俊杰 万分的舍不得，希望还有机会再见面 嘉伦的刘海的微博视频'
        line = '反正我觉得好看 罪恶天使forget的秒拍视频'
        line = '佩服这样的你 杨蓉 ... 全文'

        clean_sentence(line)


def double(matched):
    value = int(matched.group('value'))
    return str(value * 2)

if __name__ == '__main__':
    way = '1'
    read_data(way)

