import sys


def read_ids(filename):
    
    ids = set()
    with open(filename) as f:
        for line in f:
            line_id = line.strip()
            if len(line_id) > 0:
                ids.add(line_id)
    return ids
        

if __name__ == '__main__':
    
    if len(sys.argv)<4:
        print "Usage: %s <coinco-filename> <dev-ids-filename> <test-ids-filename> <eval|gold>" % sys.argv[0]
        sys.exit(1)
        
    coinco_all = open(sys.argv[1],'r')
    coinco_dev = open(sys.argv[1]+'.dev', 'w')
    coinco_test = open(sys.argv[1]+'.test', 'w')
    dev_ids = read_ids(sys.argv[2])
    test_ids = read_ids(sys.argv[3])
    format = sys.argv[4]
    
    
    '''
    eval format:    mission.N       4       1       a mission to end a war
    gold format:    mission.N 4 :: task 2;plan 2;
    '''
    for line in coinco_all:
        if len(line.strip()) > 0:
            if format == 'eval':
                line_id = line.split('\t')[1]
            elif format == 'gold':
                line_id = line.split('::')[0].strip().split()[-1]
            else:
                raise Exception('input format unknown: ' + format)
            if line_id in dev_ids:
                coinco_dev.write(line)
            elif  line_id in test_ids:
                coinco_test.write(line)
            else:
                print "NOTICE: id {} is neither in dev nor in test".format(line_id)
        
    coinco_all.close()
    coinco_dev.close()
    coinco_test.close()
