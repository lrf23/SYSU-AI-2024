import sys

def default(str):
    return str + ' [Default: %default]'

def readcommand(argv):
    """首先定义提示文本，然后创建parser解析器，之后添加可选项，然后用parse_args对参数进行解析，
    结果存在第一个返回值options,最后调用即可"""
    from optparse import OptionParser
    usageStr = """a study of the command line"""

    parser=OptionParser(usageStr)
    parser.add_option('-n', '--numGames', dest='numGames', type='int',help=default('the number of GAMES to play'), metavar='GAMES', default=3)
    parser.add_option('-g','--goal',dest='goal',type='string',help=default('the goal of the game'),metavar='GOALS',default='eat all the food')
    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    args = dict()
    args['numGames']=options.numGames
    args['goal']=options.goal
    return args
if __name__== "__main__":
    args=readcommand(sys.argv[1:])
    print(args['numGames'])
    print(args['goal'])
    print('this is %d'%3)