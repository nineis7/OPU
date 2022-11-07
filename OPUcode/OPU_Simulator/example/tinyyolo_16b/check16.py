import numpy as np
import os
import sys

def compare_result(name1, name2):
    def hex16to8(x): # 16b hex str -> 8b hex str
        xn = x[:-2]
        residue = 1 if int(x[-2:], 16) > 127 else 0
        value = int(xn, 16) + residue
        bstr = bin(value)[2:]
        if len(bstr) < 8:
            if value <= 127:
                bstr = '0' + bstr
            else:
                bstr = '1' + bstr
        elif len(bstr) > 8:
            bstr = bstr[-8:]
        xn = hex(int(bstr, 2))[2:]
        return xn
    f = open(name1)
    lines1 = f.readlines()
    f.close()
    f = open(name2)
    lines2 = f.readlines()
    f.close()
    assert(len(lines1) == len(lines2))
    diff_stats = {}
    tot = 0
    for i in range(len(lines2)):
        dat1 = [int(x, 16) for x in lines1[i].strip().split(' ')][-32:]
        dat2 = [int(hex16to8(x), 16) for x in lines2[i].strip().split(' ')]
        for j in range(len(dat2)):
            tot += 1
            diff = np.abs(dat1[j] - dat2[j])
            if not diff in diff_stats:
                diff_stats[diff] = 0
            diff_stats[diff] += 1
            '''
            if not dat1[j]==dat2[j]:
                print('line',i+1,'(',j+1,')',hex(dat1[j]),hex(dat2[j]),'val:',dat1[j],dat2[j])
                print(lines1[i].strip().split(' ')[j],lines2[i].strip().split(' ')[j])
                return
            '''
    print('diff dist (sort by count): ')
    diff_stats = dict(sorted(diff_stats.items(), key=lambda item: item[1]))
    for diff, count in diff_stats.items():
        print('\t', diff, ':', count / tot * 100, '%')
    #print('PASS')

if __name__ == "__main__":
    compare_result(sys.argv[1], sys.argv[2])

