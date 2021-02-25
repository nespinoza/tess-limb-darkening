def read_data(fname):
    fin = open(fname, 'r')
    data = {}
    while True:
        line = fin.readline()
        if line != '':
            if line[0] != '#':
                lv = line.split()
                name, ticid = lv[0], lv[1]
                data[name] = {}
                data[name]['ticid'] = ticid
        else:
            break
    return data
