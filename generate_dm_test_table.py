from model.lab import DMTest

if __name__ == '__main__':
    dm_test_object = DMTest()
    for L in ['1W', '1M', '6M']:
        dm_test_object.L = L
        dm_test_object.table()
