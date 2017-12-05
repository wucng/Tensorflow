from datetime import datetime

start=datetime.now()

import pyximport
pyximport.install()

import test7


print(datetime.now()-start)

