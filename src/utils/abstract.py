def write_abstract(args, path, stocks, start_date, end_date):
    file = path + '/abstract.txt'
    arg_dict = vars(args)
    
    with open(file, 'w') as f:
        f.write('Target stocks: ' + str(stocks) + '\n')
        f.write('Train start date: ' + start_date + '\n')
        f.write('Train end date: ' + end_date + '\n')
        
        for key, value in zip(arg_dict.keys(), arg_dict.values()):
            f.write(key + ' = ' + str(value) + '\n')