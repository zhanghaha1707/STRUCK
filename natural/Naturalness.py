import argparse
import sys
sys.path.append("..")
from utils.basic_setting import *
from utils.tools import *
from dataset.dataClass import *
from config.support import *
from struck.attacker import NewStruct_attack
from ModelHandler.attacker_baseline import *
from struck.get_code_info import *
def filter_code(min_len,max_len,save_path='filtered_id.pkl',restore=False):
    # 1. filters the codes with less than 200 tokens in test
    if not os.path.isfile(save_path) or restore:
        vocab=VocabModel()
        config=myConfig().config
        test=LSTMDataset(vocab,'test',restore=False,config=config)
        filtered_id=[]
        for i,code in enumerate(test.data):
            if min_len<len(code.tokens)<max_len:
                filtered_id.append(i)
        
        save_pkl(filtered_id,save_path)
    else:
        filtered_id=read_pkl(save_path)
    print(f"There are {len(filtered_id)} code fragments that meet the requirements, stored in filtered_id.pkl")
    return filtered_id
def write_code(model_name,save_name,ori,struck,insert,rw):
    
    path=os.path.join('./code',model_name)
    with open(os.path.join(path,'ori',f"{save_name}.c",'w')) as ori_file:
        ori_file.write(ori)
    with open(os.path.join(path,'struk',f"{save_name}.c",'w')) as struck_file:
        struck_file.write(struck)
    with open(os.path.join(path,'insert',f"{save_name}.c",'w')) as insert_file:
        insert_file.write(insert)
    with open(os.path.join(path,'random_insert',f"{save_name}.c",'w')) as rw_file:
        rw_file.write(rw)
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_model', type=str,
                        default='lstm', help="support lstm,ggnn_simple")
    parser.add_argument('--log_file_path',type=str,default="./log",help='The specially specified log to get the comparison code')
    parser.add_argument('--log_write_mode',type=str,default="a",help='log write type')
    args = parser.parse_args()
    attack_model_name = args.attack_model
    # 1. filter code
    filtered_code_path="filtered_id.pkl"
    filtered_id=filter_code(100,200,filtered_code_path)
    # 2.basic init
    set_seed()
    mycfg = myConfig()
    config = mycfg.config
    logger = logging.getLogger(__name__)
    log_path=os.path.join(args.log_file_path,f"{args.attack_model}.log")
    set_attack_logger(
            logger, log_path=log_path, write_mode='w')

    # gpu
    n_gpu, device = set_device(config)

    # 3. set target model
    
    model_config = config[attack_model_name]
    model_config['device'] = device
    vocab = vocab_map[attack_model_name](config=config)
    
    model = support_model[attack_model_name](**model_config)  
    logger.info(f"load model parameter from {model_config['load_path']}")
    model_parameter=torch.load(model_config['load_path'])
    if n_gpu in [1,2]:# GPU
        if list(model_parameter.keys())[0].startswith('module'):
            model = torch.nn.DataParallel(model,device_ids=[0, 1])
            model.load_state_dict(model_parameter)
        else:
            model.load_state_dict(model_parameter)
    else: # cpu
        model.load_state_dict(torch.load(
            model_parameter, map_location='cpu'))  # cpu上
    if n_gpu > 1 and not isinstance(model,torch.nn.DataParallel):# multi GPU
        model = torch.nn.DataParallel(model,device_ids=[0, 1])
    model.eval()
    # 3. prepare data
    data_path='../data/test/test_raw.pkl'
    dataset=read_pkl(data_path)
    # 3.Attack mode preparation
    # 3.1 struck
    struck_config = config[attack_map['new_struct']]
    struck_attack = NewStruct_attack(dataset=None,
            model=model, vocab=vocab, device=device, logger=logger, args=args, **struck_config)
    # 3.2 insert
    insert_config = config[attack_map['insert']]
    insert_args=copy.deepcopy(args)
    insert_args.attack_way='insert'
    insert_attack=StructAttack(dataset=None,model=model,loss_func=None,vocab=vocab,device=device,logger=logger,args=insert_args,**insert_config)
    # 3.3 random_insert
    rw_insert_config = config[attack_map['random_insert']]
    rw_insert_args=copy.deepcopy(args)
    rw_insert_args.attack_way='insert'
    rw_insert_attack=StructAttack(dataset=None,model=model,loss_func=None,vocab=vocab,device=device,logger=logger,args=rw_insert_args,**rw_insert_config)
    
    model_predict_succ_ids=[]
    attack_succ_all_ids=[]
    for i in filtered_id:
        code_id = dataset['id'][i]
        code = dataset['code'][i]
        label = dataset['label'][i]
        
        target_codefeatures=codefeatures_map[attack_model_name](code_id=code_id,code=code,label=label,vocab=vocab)
        is_right, old_pred, old_prob = predicti_map[attack_model_name](model, vocab, target_codefeatures, device)
        if not is_right:
            logger.info(f'The {i} th code model predicted error, not considered')
            continue
        save_name=f"{str(label)}-{str(code_id)}"
        # struck 攻击
        is_struck_succ, struck_adv_code, struck_adv_label, struck_result_type,struck_invotions,struck_gen_adv_nums,struck_changes_nums=struck_attack.attack_single_code(code, label,False)
        if struck_result_type==1:# 
            logger.info(f'struck attack {save_name} succ')
            is_insert_succ, insert_adv_codefeatures, insert_adv_label,insert_result_type,insert_invotions,insert_gen_adv_nums,insert_changes_nums=insert_attack.attack_single_code(target_codefeatures,False)
            if insert_result_type==1:
                logger.info(f'isnert attack {save_name} succ')
                is_RWnsert_succ, RWinsert_adv_codefeatures, RWinsert_adv_label,RWinsert_result_type,RWinsert_invotions,RWinsert_gen_adv_nums,RWinsert_changes_nums=rw_insert_attack.attack_single_code(target_codefeatures,False)
                if RWinsert_result_type==1:
                    
                    logger.info(f'random_insert attack {save_name} succ')
                    logger.info(f"{save_name} code three forms are successful attack, write the corresponding file")
                    write_code(attack_model_name,save_name,ori=code,struck=struck_adv_code,insert=insert_adv_codefeatures.code,rw=RWinsert_adv_codefeatures.code)
