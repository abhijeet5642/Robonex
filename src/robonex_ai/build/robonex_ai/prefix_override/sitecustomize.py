import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/abhinav/Desktop/roboControl/src/robonex_ai/install/robonex_ai'
