import time

local_a ='global'

def fun1():
    local_a = 1
    time.sleep(1)
    if local_a == 1:
        print('HELLO')

def fun2():
    time.sleep(1)

def fun3():
    time.sleep(2)

def fun4():
    time.sleep(1)

def fun5():
    time.sleep(1)
    fun4()

def Func6():
    print('123')

fun1()
fun2()
fun3()
fun5()
fun
