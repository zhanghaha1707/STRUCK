import sys 
sys.path.append('..')
from utils.basic_setting import *
from utils.tools import save_data
# remove comment
def remove_comment(text):
    '''
    description: Remove comments from code
    param text [*] code
    return [*]
    '''
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " 
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


# Divide the data set and validation set => store the format code, label, then store, and then process the data set in accordance with this, keep consistent
def split_raw_dataset(dir='./raw_data',src='ProgramData',ratio='3.2:0.8:1',save_path='./',logger=None):
    '''
    description: A file that divides and stores the original data set
    '''
    
    data={"id":[],"code":[],"label":[]}
    for label in tqdm(sorted(os.listdir(os.path.join(dir, src)))):
        for file in sorted(os.listdir(os.path.join(dir, src, label))):
            try:
                with open(os.path.join(dir, src, label, file), 'r', encoding='latin1') as _f:
                    code = _f.read()
                    code = remove_comment(code)
                    with open(os.path.join(dir, src, label, file), 'w', encoding='latin1') as _f:
                        _f.write(code)
                data['id'].append(int(file[:-4]))
                data['code'].append(code)
                data['label'].append(int(label)-1)
            except Exception as e:
                    pass       
    data_num=len(data['code'])
    rand_idx=random.sample(range(data_num), data_num)    
    ratios = [float(r) for r in ratio.split(':')]
    train_split = int(ratios[0]/sum(ratios)*data_num)
    val_split = train_split + int(ratios[1]/sum(ratios)*data_num)
    train={"id":[],"code":[],"label":[]}
    dev={"id":[],"code":[],"label":[]}
    test={"id":[],"code":[],"label":[]}# id可以
    for i in rand_idx[:train_split]:
        train["id"].append(data['id'][i])
        train['code'].append(data['code'][i])
        train['label'].append(data['label'][i])
    for i in rand_idx[train_split:val_split]:
        dev["id"].append(data['id'][i])
        dev['code'].append(data['code'][i])
        dev['label'].append(data['label'][i])
    for i in rand_idx[val_split:]:
        test["id"].append(data['id'][i])
        test['code'].append(data['code'][i])
        test['label'].append(data['label'][i])
    save_data(save_path,train,"train",True) 
    save_data(save_path,dev,"dev",True)
    save_data(save_path,test,"test",True)  

if __name__ == "__main__":
    set_seed(2022)
    split_raw_dataset()