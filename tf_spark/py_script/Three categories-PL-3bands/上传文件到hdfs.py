import os
os.popen('hdfs dfs -put ' +'mnist_name.txt'+ ' hdfs://dm01-08-01.tjidc.dcos.com:8020/user/root/mnist')

# f=os.open('hdfs://dm01-08-01.tjidc.dcos.com:8020/user/root/mnist/mnist_name.txt',os.O_APPEND)
# os.write(f,"123")
# os.close(f)

# f=open('hdfs://dm01-08-01.tjidc.dcos.com:8020/user/root/mnist/mnist_name.txt','a')
# f.write('123')
# f.close()
