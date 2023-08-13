
from pycparser.c_parser import CParser
from pycparser.c_ast import  NodeVisitor
from pycparser.c_ast import *
from sklearn.feature_selection import SelectKBest
sys.path.append("..")
from dataset.vocabClass import VocabModel
from data.vocab.forbidden import *
from utils.pattern import *
from utils.tools import * 
import copy
import string 
from utils.basic_setting import *
from collections import Counter




class valid_node(object):
    """Variable names that can be attacked"""
    def __init__(self,name=None,node_type=['var'],data_type=None,whole_name=None,prefix=[],value=None,pos=None,tokensIndents=None) -> None:
       
        self.name=name
        self.node_types=[node_type] 
        self.data_types=[data_type] 
        if whole_name is None: whole_name=name 
        self.whole_names=[whole_name]
        self.prefixs=[prefix] 
        self.values=[value] 
        self.poses=[pos]
       
        self.indents=[tokensIndents[pos]]
        self.decl_poses=[pos] 
       
        self.groups={self.get_func_prefix(prefix):[pos]}
        self.weight=1
        # =========The addSubGraph phase leveraged the stored auxiliary information===============
        self.can_redundant_values_pos={}
        self.can_copy_pos=[]
        self.can_changes_pos={}
        self.can_gen_condition_value_pos_and_can_add_pos={} 
        self.cand_condition_and_pose=[]
        self.can_new_decl_add_pose=[]
    def update_node(self,node_type=None,data_type=None,whole_name=None,prefix=[],value=None,pos=None,tokensIndents=None,is_decl=False):
       
        
        if node_type is not None:
            self.node_types.append(node_type)
        else: 
            self.node_types.append(self.node_types[-1])
        if data_type  is not None:
            self.data_types.append(data_type)
        else: 
            self.data_types.append(self.data_types[-1])
        if prefix !=[]:
            self.prefixs.append(prefix)
        else:
            self.prefixs.append(None)
        if value is not None and value !='rvalue':
            self.values.append(value)
        else:
            is_val_hold=False
            
            for i in range(len(self.prefixs)-2,-1,-1):
                if modifyStruct.cheack_prefixs(self.prefixs[i],self.prefixs[-1]):
                    is_val_hold=True
                    break
            
            if not is_val_hold or self.values[i] is None:
                self.values.append(value)# None/rvalue
            else:
                self.values.append(self.values[i])
        if whole_name is not None:
            self.whole_names.append(whole_name)
        else:
            self.whole_names.append(self.name)
        if is_decl:
            self.decl_poses.append(pos)
        self.poses.append(pos)
        self.weight+=1 
        self.indents.append(tokensIndents[pos])
        try:
            self.groups[self.get_func_prefix(prefix)].append(pos)
        except:
            self.groups[self.get_func_prefix(prefix)]=[pos]
    def updateIndents(self,tokensIndents):
       

        self.indents=[tokensIndents[pos] for pos in self.poses]
    def show_info(self,tokensPos2tokenStmts=None):
       
        info=f"\n\
            ===>name:{self.name}<=====\n\
            node_types{self.node_types}\n\
            data_types:{self.data_types}\n\
            whole_name:{self.whole_names}\n\
            prefixs:{self.prefixs}\n\
            values:{self.values}\n\
            groups:{self.groups}\n"
       
        return info
    def get_func_prefix(self,prefix):
        if prefix is None or prefix==[]:
            return None
        for pre in prefix:
            if pre.split()[0].startswith('func'):
                return pre
        return 'other'

class func_block(object):
  
    def __init__(self,name,whole_name,return_type=None,pos=None,tokensIndents=None,decl_prefix=[]) -> None:
       
        self.name=name
        self.whole_name=whole_name 
        self.types=[return_type]
        self.func_name_pos=[pos] 
        self.decl_prefix=decl_prefix
        self.def_poses=[]  
        
        self.def_vars=[] 
        self.after_line=-1
        self.def_after_line=-1
       
        self.call_poses=[] 
        self.call_vars=[]  
        self.func_name_idents=[tokensIndents[pos]]
        self.func_call_prefix=[]
        self.weight=1
    def update_def_pos(self,begin_pos,end_pos):
        
        self.def_poses.append((begin_pos,end_pos))
        self.weight+=end_pos-begin_pos+1
    def update_func_appear(self,pos=None,type=None,tokensIndents=None):
       
        if type is not None:
            self.types.append(type)
        else:
            self.types.append(self.types[-1])
        self.func_name_pos.append(pos) 
        self.weight+=1
        self.func_name_idents.append(tokensIndents[pos])
    def update_call(self,pos,using_var=None,tokensIndents=None,prefix=[]):
        
        self.call_poses.append(pos)
        self.call_vars.append(using_var)
        self.update_func_appear(pos=pos,tokensIndents=tokensIndents)
        if prefix is not None:
            if self.func_call_prefix==None:
                self.func_call_prefix=prefix
            else:
                self.func_call_prefix=modifyStruct.get_out_prefix(prefix,self.func_call_prefix)#获取函数调用最外层的prefix 
    def update_func_name_indents(self,tokensIndents):
       
        self.func_name_idents=[tokensIndents[pos] for pos in self.func_name_pos]
    def show_info(self,tokensPos2tokenStmts=None):
        info=f"\n\
            name:{self.name}\n\
            poses:{self.def_poses}\n\
            vars:{self.def_vars}\n\
            types:{self.types}\n\
            call_poses:{self.call_poses}\n"
        info+=f"\
            the function name appears positions:\n\
            prefix:{self.func_call_prefix}\n\
            The declaration of the function needs to be specified after {self.after_line}line\n"
            
        if tokensPos2tokenStmts is not None:
            for i,pos in enumerate(self.func_name_pos):
               info+=f'\
                {pos}=>loc:{tokensPos2tokenStmts[pos]}=>Indent:{self.func_name_idents[i]}\n'   
        else:
            for i,pos in enumerate(self.func_name_pos):
               info+=f'\
                {pos}=>Indent:{self.func_name_idents[i]}\n'   
        for pos,var in zip(self.call_poses,self.call_vars):
            info+=f"\
            在{pos}用参数{var}调用函数"
        return info
    
class valid_block(object):
    """block"""
    def __init__(self,name,block_type,begin_pos=None,end_pos=None,using_var=None,prefix=[],tokensIndents=None,all_tokens_nums=1) -> None:
        
        self.name=name
        self.type=block_type   
        self.begin_pos=begin_pos
        self.end_pos=end_pos   
        # if end_pos:
        self.tokens_num=end_pos-begin_pos+1
       
        self.last_if_pos=0
        self.last_if_pos_has_else=False# 
      
        self.prefix=prefix
        self.using_var=using_var
        self.Indents=tokensIndents[begin_pos]
        self.weight=self.tokens_num/all_tokens_nums
    def show_info(self,tokensPos2tokenStmts=None):
        info=f"\n\
            block type:{self.type}\n\
            block name:{self.name}\n\
            prefix:{self.prefix}\n\
            begin pos:{self.begin_pos}=>Indents:{self.Indents}\n\
            end pos:{self.end_pos}\n\
            token num:{self.tokens_num}\n\
            parameter:{self.using_var}\n"
        return info
class data_block(object):
    
    def __init__(self,data_type,name,poses=None,pos=None) -> None:
        self.data_type=data_type
        self.name=name
        self.def_poses=poses
        self.pos=[pos] 
        self.example={}
        self.var=[]
        self.first_using_pos=None
        self.first_using_Node=None
        self.after_line=-1
        self.outer_prefix=[]
        if poses is None:
            self.decl_pos=pos
            self.weight=1
        else:
            self.decl_pos=poses[0]
            self.weight=poses[1]-poses[0]+1
    def show_info(self,tokensPos2tokenStmts=None):
        info=f"\
            type:{self.data_type}\n\
            name:{self.name}\n\
            def poses:{self.def_poses}\n\
            vars:{self.var}\n\
            decl pos:{self.pos}\n\
            use pos:{self.example}\n\
            first using pos：{self.first_using_pos}\n\
            outer prefix={self.outer_prefix}\n"
        return info
class getCodeInfo(NodeVisitor):
   
    def __init__(self,ori_code=None,code_tokens=None):
         #====辅助工具======
        self.is_node=False  
        self.is_block=False 
        self.is_func=-1 
        self.is_for=-1
        self.prefix=[]
        self.data_block_num=0 
        self.cand_end_flag=[',',';']
        self.block_match={"{":"}",'"':'"',"(":")","'":"'"}
        self.block_types=['for','if','while','do']

       
        self.pose_and_body_list={}
        # ========stage2using info==========
        # ==============================init=================
        self.init_info(ori_code,code_tokens)
        self.coord2tokensPos,self.tokenStmts2tokensPos,self.tokensPos2tokenStmts=self.getCoord2toknesLoc(self.str_stmts,self.token_stmts)
        self.tokensEnd,self.stmtsEnd,self.one_line_sent=self.getTokensEnd()# self.one_line_sent 
    
        # Statistical information is required
        #=======related node=======
        
        self.data_type=[]
        self.node_type=[]
        self.init_value=None 
        self.whole_name=None
       
        self.valid_variable_names=[]
        self.valid_nodes={} 
        
        self.decl={}
        self.assign=None
        self.is_decl=False
        self.ops=[] 
        self.typedef_node={}
        self.typedef_struct={}
        #=======related node =======
        #======related block=======
        
        self.func_blocks={} 
        self.data_blocks={} 
        self.valid_blocks={}
        self.other_block_num={}
        self.blcok_name=''
        self.all_other_block_nums=0
        #======related block=======
        #=======Traversing the code to get relevant information===
        self.getInfo()
        self.update_stmtIns_pos_prefix()
    #=====Auxiliary functions======
    def init_info(self,code=None,code_tokens=None):
       
        if code_tokens is None:
            parser = pycparser.CParser()
            parser.clex.input(code)
            code_tokens=[]
            
            t = parser.clex.token()
            while t is not None:
                code_tokens.append(t.value)
                t=parser.clex.token()
        
        self.token_stmts, self.StmtInsPos_line, self.StmtIndents,self.tokensIndents=tokens2stmts(code_tokens)
        self.StmtInsPos_token,self.InsTokensIndents=getStmtInsPos_tokens(self.StmtInsPos_line,self.token_stmts)
        self.stmts_line_prefix=[[]]*len(self.token_stmts)
        self.code=stmts2code(self.token_stmts)
        
        self.str_stmts=self.code.split("\n")   
        self.code_tokens=[] 
        for tokens in self.token_stmts:
            self.code_tokens+=tokens 
        self.ori_code_tokens_nums=len(code_tokens)
    def getInfo(self):
        
        root=CParser().parse(self.code)
        self.visit(root) 
    def show_code_info(self,level=0):
        
        self.sorted_nodes=sorted(self.valid_nodes.items(),key=lambda x:len(x[1].poses),reverse=True)
        info=f"code have {len(self.valid_variable_names)} nodes can been attacked:{self.valid_variable_names}\n"
        if len(self.valid_nodes) and level in [0,1]:
            info+=f"having valid nodes {len(self.valid_nodes)} valid nodes, in order of the number of occurrences:{[key for key,_ in self.sorted_nodes]}\n"
            for node_name,item in self.sorted_nodes:
                info+=item.show_info(self.tokensPos2tokenStmts)
                
    
        if len(self.func_blocks) and level in [0,2]:
            info+=f"{len(self.func_blocks)} user-defined functions,they are{list(self.func_blocks.keys())}\n"
            for func,item in self.func_blocks.items():
                info+=item.show_info(self.tokensPos2tokenStmts)
        if len(self.data_blocks) and level in [0,3]:
            info+=f"{len(self.data_blocks)}user-defined datatypes,they are{list(self.data_blocks.keys())}\n"
            for data,item in self.data_blocks.items():
                info+=item.show_info(self.tokensPos2tokenStmts)
        if len(self.valid_blocks) and level in [0,4]:
            info+=f"{len(self.valid_blocks)} other blocks, they are{list(self.valid_blocks.keys())}\n" 
            for data,item in self.valid_blocks.items():
                info+=item.show_info(self.tokensPos2tokenStmts)
        print(info)
    def add_info_to_StmtInsPos_line(self,begin_line,end_line,prefix):
       
        last_prefix=prefix[-1]
        for line_pos in range(begin_line,end_line):
            # if else
            if last_prefix.startswith('if') and self.token_stmts[line_pos][0]=='if':
                prefix[-1]=last_prefix+'_if'
            elif last_prefix.startswith('if') and self.token_stmts[line_pos][0]=='else':
                prefix[-1]=last_prefix+'_else'
           
            if len(prefix) > len(self.stmts_line_prefix[line_pos]):
                self.stmts_line_prefix[line_pos]=prefix
        if self.token_stmts[end_line][-1]=='}' or self.StmtIndents[end_line]<self.StmtIndents[begin_line]:
            try:
                self.stmts_line_prefix[end_line]=prefix[:-1]
            except:
                self.stmts_line_prefix[end_line]=[]
        
    def update_stmtIns_pos_prefix(self):
        self.stmtIns_pos_prefix=[None]*len(self.StmtInsPos_token)
        for i,pos in enumerate(self.StmtInsPos_token):
            line=self.tokensPos2tokenStmts[pos][0]
            if self.stmts_line_prefix[line] != []:
                self.stmtIns_pos_prefix[i]=self.stmts_line_prefix[line]
    def reset_parameter(self):
        self.node_name=None
        self.data_type=[]
        self.init_value=None
        self.whole_name=None
    def getCoord2toknesLoc(self,str_stmts,tokens_stmts):
        
        
        coord2tokensPos={}
        tokenStmts2tokensPos={}
        loc_id=0
        tokens_id=0
        for line in range(len(str_stmts)):
            if str_stmts[line].startswith(" "):
                is_start_with_space=True
            else:
                is_start_with_space=False
            colum=0
            str_singal=[] 
            while colum < len(str_stmts[line]):
                coord2tokensPos[(line,colum)]=loc_id
                if str_stmts[line][colum]!=' ' and  is_start_with_space:
                    is_start_with_space=False
               
                if str_stmts[line][colum]==' ' and  not is_start_with_space and len(str_singal)==0:
                    while str_stmts[line][colum]==' ' and colum< len(str_stmts[line])-1:
                        colum+=1 
                    loc_id+=1
                    coord2tokensPos[(line,colum)]=loc_id
                
                
                if  str_stmts[line][colum] in ['"',"'"]:
                    str_len=len(self.code_tokens[loc_id])
                    for j in range(str_len):
                        coord2tokensPos[(line,colum+j)]=loc_id
                    colum+=str_len
                    continue
                    

                colum+=1
            for tokenColumnId in range(len(tokens_stmts[line])):
                tokenStmts2tokensPos[(line,tokenColumnId)]=tokens_id
                tokens_id+=1
        tokensPos2tokenStmts=[key for key in tokenStmts2tokensPos.keys()]
        return coord2tokensPos,tokenStmts2tokensPos,tokensPos2tokenStmts
    def getTokenLoc(self,coord=None,line=None,str_column=None,tokens_column=None):
        
        if coord is not None:
            return self.coord2tokensPos[(coord.line-1,coord.column-1)]
        if line is not None:
            if str_column is not None:
                return self.coord2tokensPos[(line,str_column)]
            elif tokens_column is not None:
                return self.tokenStmts2tokensPos[(line,tokens_column)]
        return None
    def getTokensEnd(self,optimize=3):
       
        heads = [stmt[0] for stmt in self.token_stmts]
        ends = [stmt[-1] for stmt in self.token_stmts]
        lens = [len(stmt) for stmt in self.token_stmts]
        n = len(ends) 
        endIndices = []
        # end token index for each line (single line statement)
        totalCnt = 0
        after_do=False
        for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, self.StmtIndents)):
            totalCnt += cnt
            if head=='do':
                after_do=True
            elif "do" in self.token_stmts[i]:
                do_indx=self.token_stmts[i].index('do')
                if do_indx<cnt-1 and end=='{':
                    after_do=True

            if end == ";":  
                if i == n-1:
                    endIndices.append(totalCnt - 1)
                elif head not in ["for", "while", "if", "else", "switch"]:
                    endIndices.append(totalCnt - 1)
                elif heads[i+1] != "else":
                    endIndices.append(totalCnt - 1)
                elif after_do and heads[i+1] == "while":#do xxx ;while() 
                    endIndices.append(None)
                    after_do=False
                else:
                    endIndices.append(None)
            elif end == "}":
                if i == n-1:
                    endIndices.append(totalCnt - 1)
                elif self.StmtIndents[i+1] < indent:
                    endIndices.append(totalCnt - 1)
                elif after_do and heads[i+1] == "while":#do{}while() 
                    endIndices.append(None)
                    after_do=False
                elif heads[i+1] != "else" :
                    endIndices.append(totalCnt - 1)
                else:
                    endIndices.append(None)
            else:
                endIndices.append(None)
        if optimize <= 0:
            res = []
            for cnt, endIdx in zip(lens, endIndices):
                res += [endIdx] * cnt 
            return res

        # end token index for each line ("for { }" & "switch { }" & "while { }" & "do { } while ();" block statement)
        totalCnt = 0
        for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, self.StmtIndents)):
            
            totalCnt += cnt
            if endIndices[i] != None:
                continue
            if head not in ["for", "while", "switch"]:
                continue
            if end == "{":
                curIdx = i + 1 
                indent-=1      
                while curIdx < n and self.StmtIndents[curIdx] > indent:
                    curIdx += 1
                assert curIdx < n and heads[curIdx] == "}", "[%d]%s"%(curIdx, heads[curIdx])
                endIndices[i] = endIndices[curIdx]
            else:# for() stmt;
                endIndices[i]=totalCnt


        totalCnt = 0
        
        for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, self.StmtIndents)):
            totalCnt += cnt
            if endIndices[i] != None:
                continue
            after_do=False
            if head in ["do"]:
                after_do=True
            elif "do" in self.token_stmts[i]:
                do_indx=self.token_stmts[i].index('do')
                if do_indx<cnt-1 and end=='{':
                    after_do=True
            if not after_do:
                continue
            if end == "{":
                curIdx = i + 1
                indent-=1
                while curIdx < n and self.StmtIndents[curIdx] > indent:
                    curIdx += 1
                assert curIdx < n and heads[curIdx] == "}", "[%d]%s"%(curIdx, heads[curIdx])
                curIdx += 1
                while curIdx < n and not (self.StmtIndents[curIdx]==indent and heads[curIdx]=="while" and ends[curIdx]==";"):
                    curIdx += 1
                
                if curIdx<n:# assert curIdx < n and heads[curIdx] == "while", "[%d]%s"%(curIdx, heads[curIdx])
                    endIndices[i] = endIndices[curIdx]

        if optimize <= 1:
            res = []
            for cnt, endIdx in zip(lens, endIndices):
                res += [endIdx] * cnt 
            return res

        # end token index for each line ("if else" statement)
        totalCnt = 0
        for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, self.StmtIndents)):
            totalCnt += cnt
            if endIndices[i] != None:
                continue
            if head not in ["if", "else"]:
                continue
            if end in ['{']:
                indent-=1
            curTotalCnt = totalCnt
            curIdx = i
            while True:
                curIdx += 1
                while curIdx < n and self.StmtIndents[curIdx] > indent:
                    curTotalCnt += lens[curIdx]
                    curIdx += 1
                assert curIdx < n   # because all single if/else statements have been processed in o-0
                if endIndices[curIdx] != None:
                    endIndices[i] = endIndices[curIdx]
                    break
        # if{} else{}
        for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, self.StmtIndents)):
            totalCnt += cnt
            # if endIndices[i] != None:
            #     continue
            if head=="}"and i+1<n and heads[i+1]=='while':# do(){}while(){}
                endIndices[i] = endIndices[i+1]
            if head=="}" and i+1<n and heads[i+1]=="else":
                endIndices[i] = endIndices[i+1]
            else:
                continue
        if optimize <= 2:
            res = []
            for cnt, endIdx in zip(lens, endIndices):
                res += [endIdx] * cnt 
            return res

        # end token index for each line (left "{ }" block statement, e.g. "int main() {}" & "enum { ...; }")
        # WARNING! This WILL occur to assertion error. NO GUARANTEE!
        totalCnt = 0
        for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, self.StmtIndents)):
            totalCnt += cnt
            if endIndices[i] != None:
                continue
            if end == "{":
                curIdx = i + 1 
                indent-=1 
                while curIdx < n and self.StmtIndents[curIdx] > indent:
                    curIdx += 1
                assert curIdx < n and heads[curIdx] == "}", "[%d]%s"%(curIdx, heads[curIdx])
                endIndices[i] = endIndices[curIdx]
        
       
        res = []
        for cnt, endIdx in zip(lens, endIndices):
            res += [endIdx] * cnt 
    
        for i,t in enumerate(self.code_tokens):
            if t in self.block_types:
                res[i]=findBlockEnd(self.code_tokens,i)
       
        one_line_sent=[]
        for line in range(len(self.token_stmts)):
            if endIndices[line] is not None:
                end_line=self.tokensPos2tokenStmts[endIndices[line]][0]
                line_content=self.token_stmts[line]
                if line==end_line and line_content[-1]==';': 
                    if line_content[0] not in ['return','}','case','if','for','switch','while','do','else','strcmp']:
                        one_line_sent.append(line)
        return res,endIndices,one_line_sent# 每个token的end和每一行的end pos
    def get_assign_from_str(self,code_str,begin_pos):
       
        end_pos=begin_pos+1
        block=[]
        while (code_str[end_pos] not in self.cand_end_flag) or len(block)!=0 :
            if len(block)==0 and code_str[end_pos]==")":# in case of for ( int i = 6 ; i <= n ; i += 2 )
                return 
            if len(block) and self.block_match[block[-1]]==code_str[end_pos]:
                block.pop(-1)
                end_pos+=1
                continue
            if code_str[end_pos] in  list(self.block_match.keys()): 
                if "'" not in block and  '"' not in block:# 特殊情况 xiao [ i ] = \'(\' ;
                    block.append(code_str[end_pos])
            if end_pos==len(code_str):
                break
            end_pos+=1
        if end_pos<len(code_str):
            assign=(code_str[begin_pos:end_pos]).strip().split(" ")
        return assign
    def get_assign_from_token(self,begin_pos):
       
        while self.code_tokens[begin_pos]!='=' and begin_pos<len(self.code_tokens):
            begin_pos+=1
        end_pos=begin_pos+1
        block=[]
        while (self.code_tokens[end_pos] not in self.cand_end_flag) or len(block)!=0 :
            if len(block)==0 and self.code_tokens[end_pos]==")":# in case of for ( int i = 6 ; i <= n ; i += 2 )
                return 
            if len(block) and self.block_match[block[-1]]==self.code_tokens[end_pos]:
                block.pop(-1)
                end_pos+=1
                continue
            if self.code_tokens[end_pos] in  list(self.block_match.keys()): 
                if "'" not in block and  '"' not in block:
                    block.append(self.code_tokens[end_pos])
            if end_pos==len(self.code_tokens):
                break
            end_pos+=1
        if end_pos<len(self.code_tokens):
            assign=self.code_tokens[begin_pos+1:end_pos]
        return " ".join(assign)
    def get_func_content(self,begin_pos):
      
        while self.code_tokens[begin_pos]!='(':
            begin_pos+=1
        end_pos=begin_pos+1
        block=[]
        while (self.code_tokens[end_pos] != ')') or len(block)!=0 :
            if len(block) and self.block_match[block[-1]]==self.code_tokens[end_pos]:
                block.pop(-1)
                end_pos+=1
                continue
            if self.code_tokens[end_pos] in  list(self.block_match.keys()): 
                if "'" not in block and  '"' not in block:
                    block.append(self.code_tokens[end_pos])
            if end_pos==len(self.code_tokens):
                break
            end_pos+=1
        if end_pos<len(self.code_tokens):
            assign=(self.code_tokens[begin_pos:end_pos+1])
        return assign
    def get_line_var(self,line):
       
        using_var=set()
        for token in line:
            if token in self.valid_variable_names:
                using_var.update([token])
        return list(using_var)
    
    def get_var_type(self,node):
        if isinstance(node,TypeDecl):
            return node.type
        elif isinstance(node,Decl):
            return self.get_var_type(node.type)
        elif isinstance(node,PtrDecl):
            return self.get_var_type(node.type)
        elif isinstance(node,Cast):
            return self.get_var_type(node.to_type)
        elif isinstance(node,Typename):
            return self.get_var_type(node.type)
    def get_line_decl_var(self,line,begin_pos,end_pos):
       
        var_nunms=1
        equals_counts=line[begin_pos:end_pos].count('=')
        comma_counts=line[begin_pos:end_pos].count(',')
       
        if comma_counts==0 or equals_counts==0 or comma_counts+1==equals_counts or '(' not in line[begin_pos:end_pos]:
            var_nunms+=comma_counts 
        else:
            
            block_begin_pos=go4next(line,'(',begin_pos)
            next_comma_pos=go4next(line,',',begin_pos)
            block_end_pos=go4match(line,'(',block_begin_pos)
            if block_begin_pos>next_comma_pos:
                var_nunms+=self.get_line_decl_var(line,next_comma_pos+1,end_pos)
            elif block_begin_pos<next_comma_pos and next_comma_pos<block_end_pos: # int b=max(a,2)-max(b,2)-max(c,d),a;
               
                next_comma_pos=go4next(line,',',block_end_pos)
                while next_comma_pos!=-1:
                    save_comma_pos=next_comma_pos
                    if '(' in line[block_end_pos:next_comma_pos]:
                        block_end_pos=go4match(line,'(',block_end_pos) 
                        if go4next(line,',',block_end_pos)==next_comma_pos:
                            break
                        else:# b=max(a,1)+max(a,1),
                            next_comma_pos=go4next(line,',',block_end_pos)
                    else:# a=max(q,1),b
                        break
                if next_comma_pos!=-1:
                    var_nunms=self.get_line_decl_var(line,begin_pos,next_comma_pos)+self.get_line_decl_var(line,next_comma_pos+1,end_pos)
        return var_nunms

            
    def get_poses_var(self,begin_pos,end_pos):
       
        begin_pos+=1
        using_var=set()
        for token in self.code_tokens[begin_pos:end_pos+1]:
            if token in self.valid_variable_names:
                using_var.update([token])
        return list(using_var)
    def get_array_dim(self,begin_pos,dim_num=None):
      
        begin_pos=go4next(self.code_tokens,'[',begin_pos)
        end_pos=go4match(self.code_tokens,'[',begin_pos)
        if self.code_tokens[end_pos+1]=='[':
            end_pos=go4match(self.code_tokens,'[',end_pos+1)
       
        return self.code_tokens[begin_pos:end_pos+1]    
    def get_whole_name(self,tokens,idx):
        
        for i in range(idx+1,len(tokens)):
            if tokens[i] in ['=',',',';']:
                return " ".join(tokens[idx:i+1])
    def get_need_name(self,node):
     
        get_name=None
        if isinstance(node,ID):
            if isinstance(node.name,str):
                get_name=node.name
            else:
                get_name=self.get_need_name(node.name)
        elif isinstance(node,UnaryOp):
            get_name=self.get_need_name(node.expr)
        elif isinstance(node,BinaryOp):
            get_name=self.get_need_name(node.left)
        elif isinstance(node,FuncCall):
            get_name=self.get_need_name(node.name)
        elif isinstance(node,StructRef):
            get_name=self.get_need_name(node.name)
        elif isinstance(node,Assignment):
           get_name=self.get_need_name(node.lvalue) 
        elif isinstance(node,ArrayRef):
            get_name=self.get_need_name(node.name)
       
        return get_name
       
    def get_struct_name(self,node):
        get_name=None
        if isinstance(node,UnaryOp):
            get_name=self.get_struct_name(node.expr)
        elif isinstance(node,Cast):
            get_name=self.get_struct_name(node.to_type)
        elif isinstance(node,Typename):
            get_name=self.get_struct_name(node.type)
        elif isinstance(node,PtrDecl):
            get_name=self.get_struct_name(node.type)
        elif isinstance(node,TypeDecl):
            get_name=self.get_struct_name(node.type)
        elif isinstance(node,IdentifierType):
            get_name=node.names[0]
        return get_name
    #======CREATE===========
    def create_valid_node(self,name,data_type=None,node_type=None,whole_name=None,prefix=[],value=None,pos=None,is_decl=None):
        
        if name in forbidden_tokens+['true','false'] and name !='ID':
            return 
        
        if prefix is None or prefix==[]:
            prefix=copy.deepcopy(self.prefix)
        if whole_name is None and self.whole_name:
            whole_name=self.whole_name
        if value is None and self.init_value is not None :
            value=self.init_value
        

        if self.valid_nodes.get(name) is None:       
            self.valid_nodes[name]=valid_node(name,copy.deepcopy(node_type),copy.deepcopy(data_type),whole_name,prefix,value,pos,self.tokensIndents)
            self.valid_variable_names.append(name)
        else: 
            if is_decl==None:
                is_decl=self.is_decl
            if pos not in self.valid_nodes[name].poses:
                self.valid_nodes[name].update_node(node_type=copy.deepcopy(node_type),data_type=copy.deepcopy(data_type),whole_name=whole_name,prefix=prefix,value=value,pos=pos,tokensIndents=self.tokensIndents,is_decl=is_decl)
        
        if data_type is None:
            if self.valid_nodes[name].data_types[-1] is not None and  self.valid_nodes[name].data_types[-1][-1] in list(self.data_blocks.keys())+list(self.typedef_node.keys()):
                try:
                    struct_name=self.typedef_node[self.valid_nodes[name].data_types[-1][-1]]
                except:
                    struct_name=self.valid_nodes[name].data_types[-1][-1]
                self.update_block_example_info(struct_name,name,pos)
        elif data_type[-1] in list(self.data_blocks.keys())+list([item for _,item in self.typedef_struct.items()]):
            try:
                struct_name=self.typedef_node[name]
            except:
                struct_name=data_type[-1]
            if self.data_blocks.get(struct_name) is not None:
                self.update_block_example_info(struct_name,name,pos)

      
        if name in self.typedef_node.keys() and self.data_blocks.get(self.typedef_node[name]) is not None:
            self.create_data_block('struct',self.typedef_node[name],pos=pos)
            self.update_block_example_info(self.typedef_node[name],name,pos=pos)
        
        self.node_type=[]
       
    def create_func_block(self,name=None,whole_name=None,return_type=None,pos=None):
        
        if name is None and whole_name is not None:
            name=whole_name.strip('*')
        if name in forbidden_tokens:
            return 
        if self.func_blocks.get(name) is None:

            self.func_blocks[name]=func_block(name,whole_name,return_type,pos,tokensIndents=self.tokensIndents,decl_prefix=self.prefix)
            self.valid_variable_names.append(name)
        else:
            self.func_blocks[name].update_func_appear(pos,return_type,tokensIndents=self.tokensIndents)
    def create_data_block(self,data_type=None,name=None,poses=None,pos=None):
       
        if self.data_blocks.get(name) is None:
            self.data_blocks[name]=data_block(data_type=data_type,name=name,poses=poses,pos=pos)
            if name in self.typedef_struct.keys() and self.valid_nodes.get(self.typedef_struct[name]) is not None:
                for pos in self.valid_nodes[self.typedef_struct[name]].poses:
                    if pos not in self.data_blocks[name].pos:
                        self.data_blocks[name].pos.append(pos)
                self.data_blocks[name].pos.sort()
            self.valid_variable_names.append(name)
           
            if poses is not None:
                self.add_info_to_StmtInsPos_line(self.tokensPos2tokenStmts[poses[0]][0],self.tokensPos2tokenStmts[poses[1]][0],copy.deepcopy(self.prefix))
            self.data_block_num+=1
        elif self.data_blocks[name].def_poses is None and poses is not None:
          
            if pos not in self.data_blocks[name].pos:
                self.data_blocks[name].pos.append(pos)
            self.data_blocks[name].def_poses=poses
            self.data_blocks[name].weight+=poses[1]-poses[0]+1
            self.add_info_to_StmtInsPos_line(self.tokensPos2tokenStmts[poses[0]][0],self.tokensPos2tokenStmts[poses[1]][0],copy.deepcopy(self.prefix))
       
        else:
            if pos not in self.data_blocks[name].pos:
                self.data_blocks[name].pos.append(pos)
    def create_valid_block(self,name,begin_pos,end_pos,using_var=[]):
       
        try:
            self.other_block_num[name]+=1
        except:
            self.other_block_num[name]=1
        self.all_other_block_nums+=1
        if isinstance(name,ID):
            name=name.name   
        block_name=name+"_"+str(self.all_other_block_nums)           
      
        self.valid_blocks[block_name]=valid_block(name=block_name,block_type=name,begin_pos=begin_pos,end_pos=end_pos,using_var=using_var,prefix=copy.deepcopy(self.prefix),tokensIndents=self.tokensIndents)    
        self.prefix.append(block_name)
        self.add_info_to_StmtInsPos_line(self.tokensPos2tokenStmts[begin_pos][0],self.tokensPos2tokenStmts[end_pos][0],copy.deepcopy(self.prefix))
        return block_name
    def update_block_example_info(self,struct_name,example_name,pos):
        try:
            if pos not in self.data_blocks[struct_name].example[example_name]:
                self.data_blocks[struct_name].example[example_name].append(pos)
                self.data_blocks[struct_name].weight+=1
        except:
            
            self.data_blocks[struct_name].example[example_name]=[pos]
            self.data_blocks[struct_name].weight+=1
        if self.valid_nodes.get(example_name) is None:
                return 
        if self.data_blocks[self.data_blcok_name].first_using_pos is None:
            can_update=False
            if self.valid_nodes[example_name].values[-1] is not None :
                can_update=True
            
            else:
                
                for pres in self.valid_nodes[example_name].prefixs:
                    if pres is None:continue
                    for pre in pres:
                        if pre.startswith('func'):
                            can_update=True
                            break
                        elif pre.startswith('struct') and struct_name not in pre:
                            can_update=True
                            break
                    if can_update:
                        break

            if can_update:
                self.data_blocks[self.data_blcok_name].first_using_pos=pos
                self.data_blocks[self.data_blcok_name].first_using_node=example_name
                self.data_blocks[self.data_blcok_name].outer_prefix=self.valid_nodes[example_name].prefixs[-1]
        elif self.valid_nodes[example_name].values[-1] is not None:
            self.data_blocks[self.data_blcok_name].outer_prefix=modifyStruct.get_out_prefix(self.data_blocks[self.data_blcok_name].outer_prefix,self.valid_nodes[example_name].prefixs[-1])
        

    #------Access functions for different types of nodes---------------
    
   
    def visit_Decl(self,node):
       

        
        
        self.data_type=[]
        #node.quals:list of storage specifiers (extern, register, etc.)
        self.data_type=self.data_type+node.quals
        # list function specifiers (i.e. inline in C99)
        self.data_type=self.data_type+node.funcspec
        # node.storage:list of storage specifiers (extern, register, etc.)
        self.data_type=self.data_type+node.storage
        
        self.init_value=None
        declpos=self.getTokenLoc(node.coord)
        self.is_decl=True
        if node.init:
            self.init_value=self.get_assign_from_token(declpos)
        else:
            self.init_value=None
        self.visit(node.type)
        self.is_decl=False
        self.init_value=None# 
        if node.init:
            self.init_value='rvalue'
            self.visit(node.init)
            self.init_value=None
    def visit_TypeDecl(self,node):   
       
        # node.quals:list of storage specifiers (extern, register, etc.)
        if " ".join(node.quals) not in  " ".join(self.data_type):
            self.data_type+=node.quals
        self.node_name = node.declname
        self.pos=self.getTokenLoc(node.coord)
        if isinstance(node.type,IdentifierType):
            
            self.data_type+=node.type.names
            if self.is_func!=0 and self.node_name:
                self.node_type.append('var')
                if self.is_func==1:
                    self.init_value='func init'
                self.create_valid_node(self.node_name,node_type=self.node_type,data_type=self.data_type,whole_name=self.whole_name,value=self.init_value,pos=self.pos) 
            self.visit(node.type)

        else:
            if isinstance(node.type,Struct):
                if node.type.name is not None:
                    self.data_blcok_name=node.type.name
                else:
                    self.data_blcok_name='struct'+str(self.data_block_num)
                self.data_type+=[self.data_blcok_name]
                self.node_type.append('struct')
                if self.node_name is not None:
                    if 'typedef' in self.data_type:
                        
                        self.typedef_node[self.node_name]=self.data_blcok_name
                        self.typedef_struct[self.data_blcok_name]=self.node_name
                    if self.is_func!=0 and self.node_name:
                        self.create_valid_node(self.node_name,data_type=self.data_type,node_type=self.node_type,pos=self.pos)
                save_data_block_name=self.data_blcok_name
                self.visit(node.type)
             
                self.data_blcok_name=save_data_block_name
                
                if node.declname is not None and node.declname not in forbidden_tokens:# 当使用结构体p1 = p2 = (struct Student *)malloc(LEN)赋值的时候 结构体没有对象declname为None
                    self.update_block_example_info(self.data_blcok_name,node.declname,self.getTokenLoc(node.coord))
                  
            if isinstance(node.type,Enum):
                if node.type.name is not None:
                    self.data_blcok_name=node.type.name
                else:
                    self.data_blcok_name='enum'+str(self.data_block_num)
                self.data_type+=[node.type.name]
                self.node_type.append('enum')
                self.create_valid_node(self.node_name,data_type=self.data_type,node_type=self.node_type,pos=self.pos)

    def visit_ArrayDecl(self, node):
        
        dim_num=1
        temp_node=node.type
        while not isinstance(temp_node,TypeDecl):
            dim_num+=1
            temp_node=temp_node.type
        if temp_node.declname is None:
            return 
        node_name = temp_node.declname
        pos=self.getTokenLoc(temp_node.coord)
        dim=self.get_array_dim(pos,dim_num)
       
        if isinstance(temp_node.type,IdentifierType): 
            self.data_type+=temp_node.type.names
            if self.is_func==1:
                self.init_value='func init'
           
            for n in temp_node.type.names:
                if self.valid_nodes.get(n) is not None:
                    struct_name=self.valid_nodes[n].data_types[0][-1]
                    if struct_name not in forbidden_tokens:
                        try:
                            self.data_blocks[struct_name].example[node_name].append(pos)
                        except:
                            self.create_data_block('struct',struct_name,pos=pos)
                            self.data_blocks[struct_name].example[node_name]=[pos]
                        break
            self.create_valid_node(node_name,node_type='arr',data_type=self.data_type,whole_name=node_name+" "+" ".join(dim),value=self.init_value,pos=pos)
        else:
            if isinstance(temp_node.type,Struct):
                if temp_node.type.name is not None:
                    self.data_blcok_name=temp_node.type.name
                else:
                    self.data_blcok_name='struct'+str(self.data_block_num)
                self.data_type+=[self.data_blcok_name]
                self.create_valid_node(node_name,node_type='struct arr',data_type=self.data_type,whole_name=node_name+" "+" ".join(dim),value=self.init_value,pos=pos)
                   
            self.visit(node.type)
            self.data_type=[]
    def visit_PtrDecl(self,node):
        temp_node=node.type
        pointer_num=1
        self.data_type+=node.quals
        while type(temp_node) not in [ArrayDecl,TypeDecl,FuncDecl]:
            self.data_type+=temp_node.quals
            temp_node=temp_node.type
            pointer_num+=1
      
        if  temp_node.coord is not None:     
            node_name_pos=self.getTokenLoc(temp_node.coord)
            node_name=self.code_tokens[node_name_pos]
            self.whole_name=" ".join(['*']*pointer_num+[node_name])
            self.node_type=['ptr']
        self.visit(temp_node)
        self.whole_name=None
    def visit_ArrayRef(self,node):
        
        pos=self.getTokenLoc(node.coord)

        if self.whole_name=='&':
            self.whole_name+=" "+" ".join([self.code_tokens[pos]]+self.get_array_dim(pos))
        else:
            self.whole_name=" ".join([self.code_tokens[pos]]+self.get_array_dim(pos))
        self.visit(node.name)
        self.visit(node.subscript) 
        self.whole_name=None
    def visit_ID(self,node):
        
        pos=self.getTokenLoc(coord=node.coord)
        if self.whole_name=='&':
            self.whole_name+=" "+node.name
        if self.data_type not in [[],None]:
            self.create_valid_node(node.name,data_type=self.data_type,whole_name=self.whole_name,pos=pos)
        else:
            self.create_valid_node(node.name,whole_name=self.whole_name,pos=pos)
        
        
    def visit_Typedef(self,node):
        
        self.node_type.append('typedef')
        # 添加限定符
        #node.quals:list of storage specifiers (extern, register, etc.)
        self.data_type=node.quals
        # 存储类node.storage:list of storage specifiers (extern, register, etc.)
        self.data_type=self.data_type+node.storage
        self.visit(node.type)
    def visit_Struct(self,node):
       
        
        if node.name is not None:
            self.data_blcok_name=node.name
            this_block_name=node.name
        else:
            self.data_blcok_name='struct'+str(self.data_block_num)
            this_block_name='struct'+str(self.data_block_num)
        self.prefix.append('struct '+self.data_blcok_name)
        pos=self.getTokenLoc(node.coord)
        if node.decls is not None:            
            end_pos=self.tokensEnd[pos]
            self.create_data_block('struct',this_block_name,(pos,end_pos),pos)
            for decl in node.decls:
                self.visit(decl)
                if decl.name not in self.data_blocks[this_block_name].var:
                    self.data_blocks[this_block_name].var.append(decl.name)
                    try:
                        
                        var_type=self.get_var_type(decl)
                        if isinstance(var_type,Struct):
                            struct_name=var_type.name
                            if struct_name != this_block_name:
                                struct_last_line=self.tokensPos2tokenStmts[self.data_blocks[struct_name].def_poses[1]][0]
                                self.data_blocks[this_block_name].after_line=max(struct_last_line,self.data_blocks[this_block_name].after_line)
                                self.update_block_example_info(struct_name,decl.name,self.getTokenLoc(decl.coord))
                        elif isinstance(var_type,IdentifierType):
                            type_name=var_type.names[0]
                            if type_name in self.typedef_node.keys():
                                typedef_line=self.tokensPos2tokenStmts[self.valid_nodes[type_name].poses[-1]][0]
                                self.data_blocks[this_block_name].after_line=max(typedef_line,self.data_blocks[this_block_name].after_line)
                    except:pass
                
        else:
            
            self.create_data_block('struct',this_block_name,pos=pos)
        self.prefix.pop(-1)   
    def visit_StructRef(self,node):
       
      
        pos=self.getTokenLoc(node.coord)
        struct_name=None
        if isinstance(node.name,ArrayRef):
            example_name=self.get_need_name(node.name)
            

            arry_dim=self.get_array_dim(pos)
            self.whole_name=" ".join(([example_name]+arry_dim+[node.type,node.field.name]))
        elif isinstance(node.name,Cast):
            example_name=node.name.expr.name
            var_type=self.get_var_type(node.name)
            if isinstance(var_type,IdentifierType):
                struct_name=var_type.names[0]
            else:
                struct_name=var_type.name
        else:
            example_name=self.get_need_name(node.name)
            if not isinstance(example_name,str):
                struct_name=self.get_struct_name(node.name)
               
            else:
                self.whole_name=" ".join([example_name,node.type,node.field.name])
        
        if isinstance(example_name,str) and self.valid_nodes.get(example_name) is not  None:
                
            for t in self.valid_nodes[example_name].data_types:
                if t is None: continue
                if t[-1].startswith('struct'):
                    struct_name=t[-1]
                    break
                elif self.data_blocks.get(t[-1]) is not None:
                    struct_name=t[-1]
                    break
                elif self.valid_nodes.get(t[-1]) is not None:
                    struct_name=self.valid_nodes[t[-1]].data_types[0][-1]
        self.visit(node.name)
        if struct_name is not None:
            if self.data_blocks.get(struct_name) is None:
                struct_name=self.valid_nodes[struct_name].data_types[0][-1]
            try:
                self.data_blocks[struct_name].example[example_name].append(self.getTokenLoc(node.name.coord))
            except:
                self.create_data_block('struct',struct_name,pos=pos)
                self.data_blocks[struct_name].example[example_name]=[self.getTokenLoc(node.name.coord)]
            self.data_blocks[struct_name].weight+=1
           
            if self.data_blocks[struct_name].first_using_pos is None:
               
                if example_name is not  None and( self.valid_nodes[example_name].values[-1] is not None or (len(self.valid_nodes[example_name].prefixs)>1 and self.valid_nodes[example_name].prefixs[0] in [[],None] and self.valid_nodes[example_name].prefixs[-1]!=self.valid_nodes[example_name].prefixs[0])):
                    self.data_blocks[struct_name].first_using_pos=self.getTokenLoc(node.name.coord)
                    self.data_blocks[struct_name].first_using_node=example_name
                    self.data_blocks[struct_name].outer_prefix=self.valid_nodes[example_name].prefixs[-1]
                elif example_name is None:
                    self.data_blocks[struct_name].first_using_pos=pos
                    self.data_blocks[struct_name].first_using_node=None
                    self.data_blocks[struct_name].outer_prefix=self.stmts_line_prefix[self.tokensPos2tokenStmts[pos][0]]
            elif example_name is not  None and self.valid_nodes[example_name].values[-1] is not None:
                self.data_blocks[struct_name].outer_prefix=modifyStruct.get_out_prefix(self.data_blocks[struct_name].outer_prefix,self.valid_nodes[example_name].prefixs[-1])
            else:
                line_prefix=self.stmts_line_prefix[self.tokensPos2tokenStmts[pos][0]]
                self.data_blocks[struct_name].outer_prefix=modifyStruct.get_out_prefix(self.data_blocks[struct_name].outer_prefix,line_prefix)
        self.visit(node.field)
        self.whole_name=None  
        
        
        

    def visit_Enum(self,node):
       
        if node.name is None:
            self.data_blcok_name='enum'+str(self.data_block_num)
            self.prefix.append('Enum '+str(self.data_block_num))
        else:
            self.data_blcok_name=node.name
            self.prefix.append('Enum '+ node.name)
        pos=self.getTokenLoc(node.coord)
        end_pos=self.tokensEnd[pos]
        self.create_data_block('enum',self.data_blcok_name,(pos,end_pos))
        for val in node.values.enumerators:
            if val.name not in self.data_blocks[self.data_blcok_name].var:
                self.data_blocks[self.data_blcok_name].var.append(val.name)
            self.visit(val)
        self.prefix.pop(-1)    
    def visit_Enumerator(self,node):
        
        node_name=node.name
        pos=self.getTokenLoc(node.coord)
        value=None
        if node.value is not None:
            value=self.get_assign_from_token(pos)
        self.create_valid_node(node_name,data_type='int',value=value,pos=pos)
    def visit_Union(self,node):
        self.prefix.append('union '+ node.name)
        pos=self.getTokenLoc(node.coord)
        if node.decls is not None:            
            end_pos=self.tokensEnd[pos]
            self.create_data_block('union',node.name,(pos,end_pos))
            for decl in node.decls:
                if decl.name not in self.data_blocks[node.name].var:
                    self.data_blocks[node.name].var.append(decl.name)
                self.visit(decl)
        self.prefix.pop(-1)
                    
    def visit_FuncDecl(self,node):
        

        if isinstance(node.type,PtrDecl):
            nums_ptr=0
            temp_node=node.type
            while isinstance(temp_node,PtrDecl):
                temp_node=temp_node.type
                nums_ptr+=1
            whole_func_name="*"*nums_ptr+temp_node.declname
            self.func_name=temp_node.declname
        else:
            self.func_name=whole_func_name=node.type.declname
        self.prefix.append('func '+self.func_name)
        self.is_func=0
        self.visit(node.type)
        try:
            data_type=node.type.name
        except:
            data_type=copy.deepcopy(self.data_type)
        self.create_func_block(self.func_name,whole_name=whole_func_name,return_type=data_type,pos=self.getTokenLoc(node.coord))
        self.is_func=1
        if node.args:
            self.visit(node.args)
        if self.func_blocks.get(self.func_name) is not None:    
            
            pos=self.getTokenLoc(node.coord)
            decl_end_pos=go4match(self.code_tokens,'(',pos)
            using_var=self.get_line_var(self.code_tokens[pos:decl_end_pos])+[data_type[-1]]
            after_line=-1
            for var in using_var:
                if var ==self.func_name:continue
                if var in self.data_blocks.keys():
                    try:
                        block_last_line=self.tokensPos2tokenStmts[self.data_blocks[var].def_poses[1]][0]
                    except:
                        block_last_line=self.tokensPos2tokenStmts[self.data_blocks[var].decl_pos][0]
                    after_line=max(after_line,block_last_line)
                    continue
                elif var in self.typedef_node.keys(): 
                    block_last_line=self.tokensPos2tokenStmts[self.valid_nodes[var].poses[0]][0]
                    after_line=max(after_line,block_last_line)
                elif var in self.valid_nodes.keys():
                    for t in self.valid_nodes[var].data_types:
                        if t is None: continue
                        if t[-1].startswith('struct') or self.data_blocks.get(t[-1]) is not None:
                            struct_name=t[-1]
                            try:
                                block_last_line=self.tokensPos2tokenStmts[self.data_blocks[struct_name].def_poses[1]][0]
                            except:
                                block_last_line=self.tokensPos2tokenStmts[self.data_blocks[struct_name].decl_pos][0]
                            after_line=max(after_line,block_last_line)
                            break  
                elif var in self.func_blocks.keys():
                    func_decl_line=self.tokensPos2tokenStmts[self.func_blocks[var].func_name_pos[0]][0]
                    after_line=max(after_line,func_decl_line)
                    continue
                
            self.func_blocks[self.func_name].after_line=after_line  
        self.is_func=-1
        self.prefix.pop(-1)
    def visit_FuncDef(self,node):
       

        self.visit(node.decl)  
        func_def_begin_pos=self.getTokenLoc(node.coord)
        func_def_end_pos=self.tokensEnd[func_def_begin_pos]
        
        if self.func_name not in forbidden_tokens:
            self.func_blocks[self.func_name].update_def_pos(func_def_begin_pos,func_def_end_pos)
        self.prefix.append('func '+self.func_name)
        self.add_info_to_StmtInsPos_line(self.tokensPos2tokenStmts[func_def_begin_pos][0],self.tokensPos2tokenStmts[func_def_end_pos][0],copy.deepcopy(self.prefix))# 确保函数调用内部的prefix
        self.visit(node.body)
        if self.func_name not in forbidden_tokens:
            using_var=self.get_poses_var(func_def_begin_pos,func_def_end_pos)
            self.func_blocks[self.func_name].def_vars=using_var
            def_after_line=-1
            for var in using_var:
                if var ==self.func_name:continue
                if var in self.data_blocks.keys():
                    try:
                        block_last_line=self.tokensPos2tokenStmts[self.data_blocks[var].def_poses[1]][0]
                    except:
                        block_last_line=self.tokensPos2tokenStmts[self.data_blocks[var].decl_pos][0]
                    def_after_line=max(def_after_line,block_last_line)
                    continue
                elif var in self.valid_nodes.keys():
                    for t in self.valid_nodes[var].data_types:
                        if t is None:continue
                        if t[-1].startswith('struct') or self.data_blocks.get(t[-1]) is not None:
                            struct_name=t[-1]
                            try:
                                block_last_line=self.tokensPos2tokenStmts[self.data_blocks[struct_name].def_poses[1]][0]
                            except:
                                
                                block_last_line=self.tokensPos2tokenStmts[self.data_blocks[struct_name].decl_pos][0]
                            def_after_line=max(def_after_line,block_last_line)
                            break    
            self.func_blocks[self.func_name].def_after_line=def_after_line
        self.func_name=None
        self.prefix.pop(-1)
    def visit_FuncCall(self,node):
        
        func_name=self.get_need_name(node)
          
        pos=self.getTokenLoc(node.coord)
        using_var=self.get_poses_var(begin_pos=pos,end_pos=self.tokensEnd[pos])
    
        if func_name in self.func_blocks.keys():
            self.func_blocks[func_name].update_call(pos,using_var,tokensIndents=self.tokensIndents,prefix=copy.deepcopy(self.prefix))
            self.prefix.append('func '+func_name)
        elif "*"+func_name in self.func_blocks.keys():
            func_name="*"+func_name
            self.func_blocks[func_name].update_call(pos,using_var,tokensIndents=self.tokensIndents,prefix=copy.deepcopy(self.prefix))
            self.prefix.append('func '+func_name)
         
        else:            
            self.create_valid_block(name=func_name,begin_pos=pos,end_pos=self.tokensEnd[pos],using_var=using_var)
            if func_name in ['scanf','get','gets']:
                self.init_value='input'
        if node.args is not None:
            self.visit(node.args)
        self.prefix.pop(-1)
        self.init_value=None
    
    def visit_Assignment(self,node):  
       
        pos=self.getTokenLoc(node.lvalue.coord)
        equal_begin_poses=[pos]
        if node.op=='=':
            if isinstance(node.rvalue,Assignment):
                temp_node=node
                try:
                    while temp_node.rvalue.op=='=':
                        temp_node=temp_node.rvalue
                        equal_begin_poses.append(self.getTokenLoc(temp_node.lvalue.coord))
                        
                except:
                    pass
            node_name=self.code_tokens[pos]
            
            equal_pos=equal_begin_poses[-1]+1 
            while self.code_tokens[equal_pos]!='=':
                equal_pos+=1
            
            
            cur_line=self.tokensPos2tokenStmts[pos][0]
            cur_line_content=self.token_stmts[cur_line]
            first_token_pos=self.tokenStmts2tokensPos[cur_line,0]
            first_equal_pos=cur_line_content[pos-first_token_pos:].index('=')
            first_equal_pos=pos+first_equal_pos
           
            cand_end_flag=[',',';','=',':']
            block_sign=[')']
            block_match={"{":"}",'"':'"',"(":")","'":"'"}
            block=[]
            
            left_val_begin_pos=pos-1
          
            for token in self.code_tokens[pos:first_equal_pos]:
                if token in block_sign:
                    block.append(token)
            
            while left_val_begin_pos>=first_token_pos:
                if self.code_tokens[left_val_begin_pos]=='(':
                    if block_match['('] in block:
                        block.remove(block_match['('])
                    else:break
                elif self.code_tokens[left_val_begin_pos] in cand_end_flag:
                    break
                elif self.code_tokens[left_val_begin_pos]==')':
                    begin_sin=gobefore4match(self.code_tokens,')',left_val_begin_pos)
                    if self.code_tokens[begin_sin-1] in self.block_types:
                        break

                left_val_begin_pos-=1
            whole_name=" ".join(self.code_tokens[left_val_begin_pos+1:first_equal_pos])

        
            assign=self.get_assign_from_token(equal_pos)
            
            
            if assign is not None and whole_name in assign:
                if modifyStruct.check_selfOp(whole_name,assign):
                    assign='selfOp'
            self.create_valid_node(name=node_name,whole_name=whole_name,value=assign,pos=pos)
        
        self.visit(node.lvalue)
        self.init_value='rvalue'
        self.visit(node.rvalue)
        self.init_value=None
           

    def visit_BinaryOp(self,node):
        
        if node.op=='>>':
            self.ops.append(node.op)
        self.visit(node.left)
       
        if node.op=='>>':
            if isinstance(node.left,ID) and node.left.name=='cin':
                self.init_value='input'
      
        if node.op=='*' and isinstance(node.left,ID) and self.data_blocks.get(node.left.name) is not None:
            self.data_type= ['struct',node.left.name]
            if isinstance(node.right,ID):
                self.whole_name='*'+node.right.name    
        self.visit(node.right)
       
        if len(self.ops)>0:
            self.ops.pop(-1)
        if len(self.ops)==0:
            self.init_value=None
    def visit_UnaryOp(self,node):
       
        if node.op.startswith('p'):
           
            self.init_value='UnaryChanged'
        if node.op=='&':
            self.whole_name='&'
       
        self.visit(node.expr) 
        if self.init_value=='UnaryChanged':
            self.init_value=None
        self.whole_name=None
    
  
    def visit_If(self,node):
       
        pos=self.getTokenLoc(node.coord)
        last_if_pos=pos
        last_if_pos_has_else=False
        temp_node=node.iffalse
        while temp_node is not None:
            last_if_pos_has_else=True
            if isinstance(temp_node,If):
                last_if_pos=self.getTokenLoc(temp_node.coord)
                temp_node=temp_node.iffalse
                if temp_node is None:
                    last_if_pos_has_else=False
            else:
                break
        
        endpos=self.tokensEnd[pos]
    
        block_name=self.create_valid_block(name='if',begin_pos=pos,end_pos=endpos)
        self.valid_blocks[block_name].last_if_pos=last_if_pos
        self.valid_blocks[block_name].last_if_pos_has_else=last_if_pos_has_else
       
        self.visit(node.cond)
        if node.iftrue is not None:
            self.visit(node.iftrue)
        if node.iffalse is not None:
            self.visit(node.iffalse)
        using_var=self.get_poses_var(begin_pos=pos,end_pos=endpos)
        self.valid_blocks[self.prefix[-1]].using_var=using_var
        self.prefix.pop(-1)
       
    def visit_For(self,node):
      
        pos=self.getTokenLoc(node.coord)
        self.create_valid_block(name='for',begin_pos=pos,end_pos=self.tokensEnd[pos])
        if node.init is not None:
            self.is_for=1
            self.visit(node.init)
        if node.cond is not None:
            self.is_for=2
            self.visit(node.cond)
        if node.next is not None:
            self.is_for=3
            self.visit(node.next)
        
        self.is_for=0
        self.visit(node.stmt)
        using_var=self.get_poses_var(begin_pos=pos,end_pos=self.tokensEnd[pos])
        self.valid_blocks[self.prefix[-1]].using_var=using_var
        self.is_for=-1
        self.prefix.pop(-1)
    def visit_While(self, node):
      
        pos=self.getTokenLoc(node.coord)
        self.create_valid_block(name='while',begin_pos=pos,end_pos=self.tokensEnd[pos])
       
        if node.cond is not None:
            self.visit(node.cond)
       
        self.visit(node.stmt)
        using_var=self.get_poses_var(begin_pos=pos,end_pos=self.tokensEnd[pos])
        self.valid_blocks[self.prefix[-1]].using_var=using_var
        self.prefix.pop(-1)
    def visit_DoWhile(self,node):
        pos=self.getTokenLoc(node.coord)
        self.create_valid_block(name='dowhile',begin_pos=pos,end_pos=self.tokensEnd[pos])
        
        if node.cond is not None:
            self.visit(node.cond)
     
        self.visit(node.stmt)
        using_var=self.get_poses_var(begin_pos=pos,end_pos=self.tokensEnd[pos])
        self.valid_blocks[self.prefix[-1]].using_var=using_var
        self.prefix.pop(-1)
    def visit_Switch(self,node):
     
        pos=self.getTokenLoc(node.coord)
        self.create_valid_block(name='switch',begin_pos=pos,end_pos=self.tokensEnd[pos])
        
        if node.cond is not None:
            self.visit(node.cond)
      
        self.visit(node.stmt)
        using_var=self.get_poses_var(begin_pos=pos,end_pos=self.tokensEnd[pos])
        self.valid_blocks[self.prefix[-1]].using_var=using_var
        self.prefix.pop(-1)
    def visit_Cast(self,node):
     
        self.visit(node.to_type)
        self.visit(node.expr)

    def visit_IdentifierType(self,node):
       
        for n in  node.names:
            if n in self.typedef_node.keys():
                self.create_valid_node(n,pos=self.getTokenLoc(node.coord))
                
    def generic_visit(self,node):
       
        for child in node.children():
            self.visit(child[1])    

class modifyStruct(object):
   
    def __init__(self,code,model=None,config=None,logger=None) -> None:
        self.codeinfo=getCodeInfo(code)
        self.config=config
       
        self.change_limits=len(self.codeinfo.code_tokens)*config['change_limts_factor']
        self.modifyMap={'if':'if','for':'while','while':'for','dowhile':'for'}
        self.changeOp={"+":"-","-":"+"}
        self.condition_head=['if ( condition ) ','while ( condition ) ']
        self.letters = string.ascii_lowercase
        self.fixed_inserts= [
        ";",
        "{ }",
        "printf ( \"\" ) ;",
        "if ( false ) ;",
        "if ( true ) { }",
        "if ( false ) ; else { }",
        "if ( 0 ) ;",
        "if ( false ) { int cnt = 0 ; for ( int i = 0 ; i < 123 ; i ++ ) cnt += 1 ; }",
        "for ( int i = 0 ; i < 100 ; i ++ ) break ;",
        "for ( int i = 0 ; i < 0 ; i ++ ) { }",
        "while ( false ) ;",
        "while ( 0 ) ;",
        "while ( true ) break ;",
        "for ( int i = 0 ; i < 10 ; i ++ ) { for ( int j = 0 ; j < 10 ; j ++ ) break ; break ; }",
        "do { } while ( false ) ;"]
       
        self.attack_way_map={'moveNode':self.moveNode_n_targets,
                            'redundantNode':self.RedundantNode_n_targets,
                            'copyNode':self.CopyNode_n_targets,
                            'changeAndRecoverNode':self.changeAndRecoverNode_n_targets,
                            'addOldDecl':self.addNodeDecl_n_targets,
                            'addNewDecl':self.addNewNode,
                            'addSubGraph':self.addSubGraph,
                            'modifyDatablock':self.modifyDatablock_n_targets,
                            'modifyFuncblock':self.modifyFuncBlock_n_targets,
                            'ReconsBlock':self.modifyblock_n_targets}
        self.stage_choose=config['stage_choose']
        if 1 in self.stage_choose:
            self.attack_stage=1 
        elif 2 in self.stage_choose:
            self.attack_stage=2
        else:
            assert 'no support attack stage'
        self.attack_without_attackDict,self.attack_with_attackDict,self.attack_infos=self.init_attack_way()
        ##===satge2 attack uing info========
        self.attackDict={"count":0,"attack_dict_paired":{}} 
        self.attack_changes_nums=0
        self.high_frequen_tokens=None
        self.gen_bound=self.config['stage2_gen_bound']

        ##=======recode tools============
       
        self.attack_record={} 
        self.operations=[]
    def init_attack_way(self):
       
        """
        0 all，
        1'moveNode', '2 modifyDatablock',' 3 modifyFuncblock', 4 ReconsBlock'
        5'redundantNode', 6'copyNode', 7 'changeAndRecoverNode', 8 'addOldDecl',9 'addNewDecl', 10  'addSubGraph'
        """
      
        all_attack_without_attackDict=['moveNode','modifyDatablock','modifyFuncblock','ReconsBlock']
        attack_without_attackDict=[]
        
        all_attack_with_attackDict=['redundantNode','copyNode','changeAndRecoverNode','addOldDecl','addNewDecl','addSubGraph']
        attack_with_attackDict=[]
        attack_choose=self.config['attack_choose']
        
        attack_infos={}
        for attack_way in all_attack_without_attackDict:
            if 'Node' in attack_way:
                if 0 in attack_choose or 1 in attack_choose:
                    
                    targets_node=list(self.codeinfo.valid_nodes.keys())
                    for _,item in self.codeinfo.data_blocks.items():
                      
                        targets_node=[node for node in targets_node if node not in item.var and node not in item.example.keys()]                    
                    if len(targets_node) >0:
                        attack_without_attackDict.append(attack_way)# moveNode
                        attack_infos[attack_way]={'times':len(targets_node)*self.config['stage1_factor'],'targets':targets_node}
            elif 'Datablock' in attack_way and len(self.codeinfo.data_blocks)!=0:
                if 0 in attack_choose or 2 in attack_choose:
                    attack_without_attackDict.append(attack_way)# modifyDatablock
                    attack_infos[attack_way]={'times':len(self.codeinfo.data_blocks)*self.config['stage1_factor'],'targets':list(self.codeinfo.data_blocks.keys())}
            elif 'Funcblock' in attack_way and len(self.codeinfo.func_blocks)!=0:
                if 0 in attack_choose or 3 in attack_choose:
                    attack_without_attackDict.append(attack_way)# modifyFuncblock
                    attack_infos[attack_way]={'times':len(self.codeinfo.func_blocks)*self.config['stage1_factor'],'targets':list(self.codeinfo.func_blocks.keys())}
            elif 'ReconsBlock' in attack_way and len(self.codeinfo.valid_blocks)!=0:
                if 0 in attack_choose or 4 in attack_choose:
                   
                    block_types={}
                    valid_block_names=[]
                    for _,block in self.codeinfo.valid_blocks.items():
                        if block.type in self.modifyMap.keys():
                            valid_block_names.append(block.name)
                            try:
                                block_types[block.type]+=1
                            except:
                                block_types[block.type]=1
                    
                    new_block_types=copy.deepcopy(block_types)
                    for block_type,num in block_types.items():
                        
                        map_type=self.modifyMap[block_type]
                        try:
                            new_block_types[map_type]+= num
                        except:
                            new_block_types[map_type]=num
                    if len(new_block_types)>0:
                        attack_without_attackDict.append(attack_way)# ReconsBlock
                        attack_infos[attack_way]={'times':sum([item for _,item in new_block_types.items()])*self.config['stage1_factor'],'targets':valid_block_names,"block_types_times":{key:item*self.config['stage1_factor'] for key,item in new_block_types.items()}}
        
        for attack_way in all_attack_with_attackDict:
            if attack_way == 'redundantNode' and (0 in attack_choose or 5 in attack_choose):
                attack_with_attackDict.append(attack_way)# redundantNode,copyNode,changeAndRecoverNode    
                attack_infos[attack_way]={'times':len(self.codeinfo.valid_nodes)*self.config['stage2_factor'],'targets':list(self.codeinfo.valid_nodes.keys()),'least_changes':4}# t=t;最少需要4个token
            elif attack_way=='copyNode' and (0 in attack_choose or 6 in attack_choose):
                attack_with_attackDict.append(attack_way)
                attack_infos[attack_way]={'times':len(self.codeinfo.valid_nodes)*self.config['stage2_factor'],'targets':list(self.codeinfo.valid_nodes.keys()),'least_changes':1} #t,t 最少需要一个token
            elif attack_way=='changeAndRecoverNode' and (0 in attack_choose or 7 in attack_choose):
                attack_with_attackDict.append(attack_way)
                attack_infos[attack_way]={'times':len(self.codeinfo.valid_nodes)*self.config['stage2_factor'],'targets':list(self.codeinfo.valid_nodes.keys()),'least_changes':12}
            elif attack_way=='addOldDecl' and (0 in attack_choose or 8 in attack_choose):
                attack_with_attackDict.append(attack_way)
                attack_infos[attack_way]={'times':len(self.codeinfo.valid_nodes)*self.config['stage2_factor'],'targets':list(self.codeinfo.valid_nodes.keys()),'least_changes':3} #char t; 最少需要一个token
            elif attack_way=='addNewDecl' and (0 in attack_choose or 9 in attack_choose):
                attack_with_attackDict.append(attack_way)
                attack_infos[attack_way]={"times":len(self.codeinfo.StmtInsPos_line)*self.config['stage2_factor'],'targets':self.config['topCommonTokens'],'least_changes':4}#添加newdecl限制的,vocab中的all_candidate_tokens t=t;
            elif attack_way=='addSubGraph' and (0 in attack_choose or 10 in attack_choose):
                
                attack_with_attackDict.append(attack_way)
                attack_infos[attack_way]={"times":len(self.codeinfo.StmtInsPos_token)*self.config['stage2_factor'],'targets':list(self.codeinfo.valid_nodes.keys()),'least_changes':6}# 最少需要6个token while(false);
       
        return attack_without_attackDict,attack_with_attackDict,attack_infos
    def modify_code(self,n_candidate=5):
       
        if self.attack_stage==1 and self.attack_without_attackDict!=[]:
            candidate_code_tokens,candidate_names,candidate_change_nums=self.stageOne(n_candidate)
            return candidate_code_tokens,self.cur_attack_way,candidate_names,candidate_change_nums

        if self.attack_stage==2 and self.attack_with_attackDict!=[]:
            candidate_attackDict,candidate_names,candidate_change_nums=self.stageTwo(n_candidate) 
            
            if candidate_attackDict is not None:
                return candidate_attackDict,self.cur_attack_way,candidate_names,candidate_change_nums
        
        return [],None,None,None
    
    def stageOne(self,n_candidate):
       
        if self.config['assign_attack_way'] != 'None': 
            self.cur_attack_way=self.config['assign_attack_way']  # redundantNode,copyNode,changeAndRecoverNode,addOldDecl,addNewDecl,addSubGraph
        else:
            self.cur_attack_way=weight_choice([(key,item["times"]) for key,item in self.attack_infos.items() if key in self.attack_without_attackDict ])
        
        gen_code_tokens,names=self.attack_way_map[self.cur_attack_way](n_candidate)
        candidate_code_tokens=[]
        candidate_change_nums=[]
        candidate_names=[]
        for code_tokens,name in zip(gen_code_tokens,names):
            if code_tokens==[] or code_tokens is None:
                
                try:
                    
                    if self.cur_attack_way not in ['modifyDatablock','modifyFuncblock']:
                        self.attack_infos[self.cur_attack_way]['targets'].remove(name)

                except:
                    pass
            else:
                for i in range(len(code_tokens)):
                    if code_tokens[i]!=[]:
                        change_nums=len(code_tokens[i])-self.codeinfo.ori_code_tokens_nums
                        
                        if self.change_limits - change_nums>0:
                            candidate_code_tokens.append(code_tokens[i])
                            candidate_names.append(name)
                            candidate_change_nums.append(change_nums)

        self.attack_infos[self.cur_attack_way]['times']-=1
        if self.attack_infos[self.cur_attack_way]['times']<=0 or len(self.attack_infos[self.cur_attack_way]['targets'])==0:
            self.attack_without_attackDict.remove(self.cur_attack_way)
        n_candidate = min(n_candidate,len(candidate_code_tokens))
        # 从中进行采样
        candisIdx = random.sample(range(len(candidate_code_tokens)), n_candidate)
        code_candidates = [candidate_code_tokens[candiIds] for candiIds in candisIdx]
        name_candidates = [candidate_names[candiIds] for candiIds in candisIdx]
        change_nums_candidates=[candidate_change_nums[candiIds] for candiIds in candisIdx]
        return code_candidates,name_candidates,change_nums_candidates
    
    def stageTwo(self,n_candidate):
        
       
        if self.config['assign_attack_way'] != 'None': 
            self.cur_attack_way=self.config['assign_attack_way']  # redundantNode,copyNode,changeAndRecoverNode,addOldDecl,addNewDecl,addSubGraph
        else:
            candi_attack_ways=random.sample(self.attack_with_attackDict,len(self.attack_with_attackDict))
            can_stage2_attack=False
            for attack_way in candi_attack_ways:
                if self.attack_infos[attack_way]['least_changes']<self.change_limits-self.attack_changes_nums:# 这种攻击方式可以进行
                    can_stage2_attack=True
                    self.cur_attack_way=attack_way
                    break
            
            if not can_stage2_attack:
                return None,None,None
       
        candidate_attackDict=[]
        candidate_names=[]
        candidate_change_nums=[]
        new_attack_dicts,choose_node_names=self.attack_way_map[self.cur_attack_way](n_candidate)
        for attack_dicts,name in zip(new_attack_dicts,choose_node_names):
            if attack_dicts==[] or attack_dicts is None:
               
                try:
                    self.attack_infos[self.cur_attack_way]['targets'].remove(name)
                except:
                    pass
            elif isinstance(attack_dicts,dict):
                change_nums=self.counts_attack_dict(attack_dicts)
                if self.change_limits - change_nums>0:
                    candidate_attackDict.append(attack_dicts)
                    candidate_names.append(name)
                    candidate_change_nums.append(change_nums)
            else:
                for i in range(len(attack_dicts)):
                    if attack_dicts[i]!=[]:
                      
                        change_nums=self.counts_attack_dict(attack_dicts[i])
                        if self.change_limits - change_nums>0:
                            candidate_attackDict.append(attack_dicts[i])
                           
                            candidate_names.append(name)
                            candidate_change_nums.append(change_nums)
        self.attack_infos[self.cur_attack_way]['times']-=1
        if self.cur_attack_way=='addNewDecl':
            targets_num=self.attack_infos[self.cur_attack_way]['targets']
        else:
            targets_num=len(self.attack_infos[self.cur_attack_way]['targets'])
        if self.attack_infos[self.cur_attack_way]['times']<=0 or (self.attack_infos[self.cur_attack_way]['targets']is not None and targets_num==0):
            self.attack_with_attackDict.remove(self.cur_attack_way)
        
       
        n_could_del = self.attackDict["count"] 
        n_candidate_del = n_could_del
        candi_attackDicts_del,remove_poses = self.remove_attack_content(n_candidate_del)
        for attackDicts_del,remove_pos in zip(candi_attackDicts_del,remove_poses):
          
            change_nums=self.counts_attack_dict(attackDicts_del)
            candidate_change_nums.append(change_nums)
            candidate_names.append(f'Remove {remove_pos}')
        n_candidate = min(n_candidate,len(candidate_attackDict))
        
        candisIdx = random.sample(range(len(candidate_attackDict)), n_candidate)
        attackDict_candidates = [candidate_attackDict[candiIds] for candiIds in candisIdx]
        name_candidates = [candidate_names[candiIds] for candiIds in candisIdx]
        change_nums_candidates=[candidate_change_nums[candiIds] for candiIds in candisIdx]
        return attackDict_candidates,name_candidates,change_nums_candidates
    @staticmethod
    def counts_attack_dict(attack_dict):
        
        change_nums=0
        for key,item in attack_dict.items():
            if key!="count":
                for add_content in attack_dict[key]:
                    change_nums+=len(add_content)
        return change_nums
    def update_codeinfo(self,new_code_tokens,choose_attack,name,change_nums):
        self.codeinfo=getCodeInfo(code_tokens=new_code_tokens)
        
        if self.attack_infos.get('ReconsBlock') is not None:
            self.update_attack_info_for_ReconsBlock()
        self.operations.append(f"{choose_attack}->{name}")        
        self.change_limits-=change_nums
    def update_attacDict(self,attacDict,choose_attack,name=None):
        self.attackDict=attacDict
        self.attack_changes_nums=self.counts_attack_dict(attacDict)
        if name is not None:
            operation=f"{choose_attack}->{name}"
        else:
            operation=f"{choose_attack}"
        self.operations.append(operation) 
    def update_attack_info_for_ReconsBlock(self):
        update_valid_block_names=[]
        for block_name,block in self.codeinfo.valid_blocks.items():
            if block.type in self.modifyMap.keys() and self.attack_infos['ReconsBlock']['block_types_times'].get(block.type) not in [0,None]:
                update_valid_block_names.append(block_name)
        self.attack_infos['ReconsBlock']['targets']=copy.deepcopy(update_valid_block_names)

    def show_attack_info(self):
       
        print(self.operations)
        print(self.codeinfo.code)
    
    def remove_attack_content(self,n_candidate=5):
       
        pos_candidates = canRemoveContentPos(self.attackDict) 
        pos_nums = len(pos_candidates)
        n_candidate=min(n_candidate,pos_nums)
        candisIdx = random.sample(range(pos_nums), n_candidate)
        pos_candidates = [pos_candidates[candiIds] for candiIds in candisIdx]
        new_attackDict = []
        remove_poses=[]
        for pos, listIdx in pos_candidates:
            _attackDict = copy.deepcopy(self.attackDict)
            if (pos,listIdx) in _attackDict["attack_dict_paired"].keys():
                pair_pos,pair_listIdx=_attackDict["attack_dict_paired"][(pos,listIdx)]
                
                if pos==pair_pos:
                    removeContentFromAttackdict(_attackDict, pos, listIdx,pair_listIdx)
                else:
                    removeContentFromAttackdict(_attackDict, pos, listIdx)
                    removeContentFromAttackdict(_attackDict, pair_pos,pair_listIdx)
            else:
                removeContentFromAttackdict(_attackDict, pos, listIdx)
            new_attackDict.append(_attackDict)
            remove_poses.append(pos)
        return new_attackDict,remove_poses
    def modifyNode(self):
       
        new_code_tokens,target_node=self.addNewNode()
        if new_code_tokens is not None:
            self.update_codeinfo(new_code_tokens)
          
            self.operations.append("add new node : "+target_node)
     
    # stage2==========RedundantNode=============  
    def RedundantNode_n_targets(self,n_candidates=5):
     
        choose_node_names=random.sample(self.attack_infos["redundantNode"]['targets'],len(self.attack_infos["redundantNode"]['targets']))
        new_attack_dicts=[]
        n_candidates+=self.gen_bound
        gen_nums=0
        for node_name in choose_node_names:
            attack_dicts,_=self.RedundantNode(node_name)
            new_attack_dicts.append(attack_dicts) 
            gen_nums+=len(attack_dicts)
            if gen_nums>n_candidates:
                break
        return new_attack_dicts,choose_node_names
    def RedundantNode(self,target_node_name=None):
       
        if target_node_name is None:
            
            target_node_name=random.choice(self.attack_infos['redundantNode']['targets'])
            
        if target_node_name in self.codeinfo.typedef_node.keys():
            return [],target_node_name
        target_node=self.codeinfo.valid_nodes[target_node_name]
        
        if target_node.can_redundant_values_pos==None:
            return [],target_node_name
        if target_node.can_redundant_values_pos=={}:
            for i,value in enumerate(target_node.values):
                if value is None or value in ['rvalue']:
                    continue
                if value =='func init' and 'func main' in target_node.prefixs[i]:
                    continue
                if value =='func init' and '[' in target_node.whole_names[i]:
                    continue
                if '[ ]' in  target_node.whole_names[i] or '[]' in  target_node.whole_names[i] or '&' in  target_node.whole_names[i] or '*' in  target_node.whole_names[i]:# 防止对&进行冗余化
                    continue
                target_node.can_redundant_values_pos[i]=[] 
            if target_node.can_redundant_values_pos=={}:
                target_node.can_redundant_values_pos=None
                return [],target_node_name
        attackDicts=[]
        fail_value_pos=[]
        for pos_indx in target_node.can_redundant_values_pos.keys():
            attak_dicts=self.RedundantNode_at_valuse_pos(target_node,pos_indx)
            if attak_dicts is None or len(attak_dicts)==0:
                
                fail_value_pos.append(pos_indx)
                continue
            for attack_dict in attak_dicts:
                attackDicts.append(attack_dict)
        
        for pos_indx in fail_value_pos:
            del target_node.can_redundant_values_pos[pos_indx]
        if target_node.can_redundant_values_pos=={}:
            target_node.can_redundant_values_pos=None
        return attackDicts,target_node_name
        
    def RedundantNode_at_valuse_pos(self,target_node,target_pos_indx):
        if target_node.can_redundant_values_pos[target_pos_indx] ==[]:
            
            tokens=self.codeinfo.code_tokens
            
            choos_pos=target_node.poses[target_pos_indx]
            cur_prefix=target_node.prefixs[target_pos_indx]
            try:
                next_pos=target_node.poses[target_pos_indx+1]
            except:
                next_pos=len(tokens)
            
            can_add_pos=[]
            for add_pos in range(choos_pos+1,next_pos):
                if add_pos in self.codeinfo.StmtInsPos_token:
                    pos_prefix=self.codeinfo.stmtIns_pos_prefix[self.codeinfo.StmtInsPos_token.index(add_pos)]
                    if self.cheack_prefixs(cur_prefix,pos_prefix):
                        can_add_pos.append(add_pos)
            if len(can_add_pos)==0:
                return None
            else:
               
                target_node.can_redundant_values_pos[target_pos_indx]=can_add_pos
        
        exp=self.generateNodeRedundant((self.change_limits-self.attack_changes_nums),(self.change_limits-self.attack_changes_nums)*0.8)
        new_attackDict=[]
       
        whole_name=target_node.whole_names[target_pos_indx]
        add_code=exp.replace('node',whole_name).split()# token list
        
        for add_pos in target_node.can_redundant_values_pos[target_pos_indx]:
            _attackDict = copy.deepcopy(self.attackDict)
            is_succ=InsAattack(_attackDict,add_pos,add_code)
            if is_succ:
                new_attackDict.append(_attackDict)
            return new_attackDict
    # stage2==========RedundantNode=============
    # stage2==========CopyNode=============
    def CopyNode_n_targets(self,n_candidates=5):
        
        choose_node_names=random.sample(self.attack_infos["copyNode"]['targets'],len(self.attack_infos['copyNode']['targets']))
        new_attack_dicts=[]
        n_candidates+=self.gen_bound
        gen_nums=0
        for node_name in choose_node_names:
            attack_dicts,_=self.CopyNode(node_name)
            new_attack_dicts.append(attack_dicts) 
            if attack_dicts is not None:
                gen_nums+=len(attack_dicts)
            if gen_nums>n_candidates:
                break
        return new_attack_dicts,choose_node_names
    def CopyNode(self,target_node_name=None):
       
        if target_node_name is None:
           
            target_node_name=random.choice(self.attack_infos["copyNode"]['targets'])
        
        if target_node_name in self.codeinfo.typedef_node.keys():
            return None,None
        target_node=self.codeinfo.valid_nodes[target_node_name]
        
       
        if target_node.can_copy_pos==[]:
            for choose_pos in target_node.decl_poses:
                choose_line,line_colum=self.codeinfo.tokensPos2tokenStmts[choose_pos]
                
                if '(' in self.codeinfo.token_stmts[choose_line][:line_colum]:
                    left_bracket=go4before(self.codeinfo.code_tokens,'(',choose_pos)
                    if go4match(self.codeinfo.code_tokens,'(',left_bracket)>line_colum:
                        continue
                if 'case' in self.codeinfo.token_stmts[choose_line][:line_colum]:continue
                indx=target_node.poses.index(choose_pos)
               
                choose_prefix=target_node.prefixs[indx]
                can_copy=True
                if choose_prefix in [None,[]]:continue
                for pre in choose_prefix:
                    if 'func' in pre:
                        func_name=pre.split()[-1]
                        if func_name in self.codeinfo.token_stmts[choose_line]:
                            can_copy=False
                            break
                    if 'struct' in pre:
                        can_copy=False
                        break
                if can_copy:           
                    target_node.can_copy_pos.append(choose_pos)
            if target_node.can_copy_pos==[]: 
                target_node.can_copy_pos=None
                return None,None
        elif target_node.can_copy_pos==None:
            return None,None
       
        attackDicts=[]
        for choose_pos in target_node.can_copy_pos:
            
            choose_pos_indx=target_node.poses.index(choose_pos)
            
            whole_name=target_node.whole_names[choose_pos_indx]
            _attackDict = copy.deepcopy(self.attackDict)
           
            copy_pos=choose_pos
            for copy_pos in range(choose_pos,len(self.codeinfo.code_tokens)):
                if self.codeinfo.code_tokens[copy_pos] in [',',';',':']:
                    copy_pos=copy_pos-1
                    break
            is_succ=InsAattack(_attackDict,copy_pos,[',']+whole_name.split())
            if is_succ:
                attackDicts.append(_attackDict)
          

        return attackDicts,target_node_name
    # stage2==========CopyNode=============
    # stage2==========changeAndRecoverNode=============  
   
    def changeAndRecoverNode_n_targets(self,n_candidates=5):
        
        choose_node_names=random.sample(self.attack_infos["changeAndRecoverNode"]['targets'],len(self.attack_infos["changeAndRecoverNode"]['targets']))
        new_attack_dicts=[]
        n_candidates+=self.gen_bound
        gen_nums=0
        for node_name in choose_node_names:
            attack_dicts,_=self.changeAndRecoverNode(node_name)
            new_attack_dicts.append(attack_dicts) 
            gen_nums+=len(attack_dicts)
            if gen_nums>n_candidates:
                break
        return new_attack_dicts,choose_node_names
    def changeAndRecoverNode(self,target_node_name=None):

        if target_node_name is None:

            target_node_name=random.choice(self.attack_infos['changeAndRecoverNode']['targets'])
        if target_node_name in self.codeinfo.typedef_node.keys():
            return [],target_node_name
        target_node=self.codeinfo.valid_nodes[target_node_name]

        if target_node.can_changes_pos=={}:
            for front_pos_indx in range(len(target_node.poses)-1):

                data_type=target_node.data_types[front_pos_indx]
                if data_type is None: break
                if  'int' not in data_type and 'char' not in data_type and 'float' not in data_type and 'double' not in data_type:
                    break 
                front_prefix=target_node.prefixs[front_pos_indx]
                back_prefix=target_node.prefixs[front_pos_indx+1]
                front_whole_name=target_node.whole_names[front_pos_indx]
                back_whole_name=target_node.whole_names[front_pos_indx+1]
                if self.cheack_prefixs(front_prefix,back_prefix) and front_whole_name==back_whole_name:
                    if '[ ]' not in front_whole_name:
                       target_node.can_changes_pos[front_pos_indx]=[]
            if target_node.can_changes_pos=={}:
                target_node.can_changes_pos=None
                return [],target_node_name
        
        elif target_node.can_changes_pos is None:
            return [],target_node_name
        attackDicts=[]

        fail_indx=[]
        for front_pos_indx in target_node.can_changes_pos.keys():
            attack_dicts=self.changeAndRecoverNode_at_assin_pos_indx(target_node,front_pos_indx)
            if attack_dicts==[]:
                fail_indx.append(front_pos_indx)
                continue
            for attack_dict in attack_dicts:
                attackDicts.append(attack_dict)
        for indx in fail_indx:
            del target_node.can_changes_pos[indx]
        if  target_node.can_changes_pos is None:
            return [],target_node_name
        return attackDicts,target_node_name
    def changeAndRecoverNode_at_assin_pos_indx(self,target_node,assin_pos_indx=None):
        
        front_pos=target_node.poses[assin_pos_indx]
        if target_node.can_changes_pos[assin_pos_indx] ==[]:
            back_pos=target_node.poses[assin_pos_indx+1]
            choose_poses=[]
            for pos in range(front_pos+1,back_pos):
                if pos in self.codeinfo.StmtInsPos_token:
                    choose_poses.append(pos)
            if len(choose_poses)==0:
                return []

            
            for change_pos_indx in range(len(choose_poses)):
                change_pos=choose_poses[change_pos_indx]
                for recover_pos_indx in range(change_pos_indx,len(choose_poses)):
                    recover_pos=choose_poses[recover_pos_indx]
                    
                    change_pos_prefix=self.codeinfo.stmtIns_pos_prefix[self.codeinfo.StmtInsPos_token.index(change_pos)]
                    recover_pos_prefix=self.codeinfo.stmtIns_pos_prefix[self.codeinfo.StmtInsPos_token.index(recover_pos)]
                    if self.strict_cheack_prefixs(change_pos_prefix,recover_pos_prefix):     
                        change_line=self.codeinfo.tokensPos2tokenStmts[change_pos][0]
                        recover_line=self.codeinfo.tokensPos2tokenStmts[recover_pos][0]
                        if change_line==recover_line and 'for' in self.codeinfo.token_stmts[recover_line]:
                            continue
                        target_node.can_changes_pos[assin_pos_indx].append((change_pos,recover_pos))
            if target_node.can_changes_pos[assin_pos_indx]==[]:
                return []
        attack_dicts=[]
        for (change_pos,recover_pos) in target_node.can_changes_pos[assin_pos_indx]:
            
            change_number=random.randint(0,100)
            change_op=random.choice(list(self.changeOp.keys()))
            front_whole_name=target_node.whole_names[target_node.poses.index(front_pos)]
            front_whole_name=front_whole_name.split()
            change_line_content=front_whole_name+['=']+front_whole_name+[change_op,str(change_number),';']
            recover_line_content=front_whole_name+['=']+front_whole_name+[self.changeOp[change_op],str(change_number),';']
            _attackDict = copy.deepcopy(self.attackDict)
            is_succ_change,change_attack_pos=InsAattack(_attackDict,change_pos,change_line_content)
            is_succ_recover,recover_attack_pos=InsAattack(_attackDict,recover_pos,recover_line_content)
            if is_succ_change and is_succ_recover:
                _attackDict["attack_dict_paired"][change_attack_pos]=recover_attack_pos
                _attackDict["attack_dict_paired"][recover_attack_pos]=change_attack_pos
                attack_dicts.append(_attackDict)
        return attack_dicts
    # stage2==========changeAndRecoverNode=============
    # stage2==========addOldNode=============
    def addNodeDecl_n_targets(self,n_candidates=5):
        
        choose_node_names=random.sample(self.attack_infos["addOldDecl"]['targets'],len(self.attack_infos["addOldDecl"]['targets']))
        new_attack_dicts=[]
        n_candidates+=self.gen_bound
        gen_nums=0

        for node_name in choose_node_names:
            attack_dicts,_=self.addNodeDecl(node_name)
            new_attack_dicts.append(attack_dicts)
            gen_nums+=len(attack_dicts)
            if gen_nums>n_candidates:
                break
        return new_attack_dicts,choose_node_names

    def addNodeDecl(self,target_node_name=None):

        if target_node_name is None:
            target_node_name=random.choice(self.attack_infos['addOldDecl']['targets'])
        if target_node_name in self.codeinfo.typedef_node.keys():
            return [],target_node_name
        target_node=self.codeinfo.valid_nodes[target_node_name]
        if target_node.can_new_decl_add_pose==[]:
            min_indent=min(target_node.indents)
            if min_indent==0: 
                target_node.can_new_decl_add_pose=None
                return [],target_node_name
            can_add_poses=[-1]
            for i in self.codeinfo.StmtInsPos_line:
                if self.codeinfo.StmtIndents[i]<min_indent:
                    pos=self.codeinfo.tokenStmts2tokensPos[i,len(self.codeinfo.token_stmts[i])-1]
                    can_add_poses.append(pos)
                else:
                    cur_line_prefix=self.codeinfo.stmts_line_prefix[i]
                    prefix_succ=True
                    for pre in target_node.prefixs:
                        if self.cheack_prefixs(cur_line_prefix,pre):
                            prefix_succ=False
                            break
                    if prefix_succ:
                        pos=self.codeinfo.tokenStmts2tokensPos[i,len(self.codeinfo.token_stmts[i])-1]
                        can_add_poses.append(pos)
            if len(can_add_poses)==0:
                target_node.can_new_decl_add_pose=None
                return [],target_node_name
            else:
                target_node.can_new_decl_add_pose=can_add_poses
        if target_node.can_new_decl_add_pose is None:
            return [],target_node_name
        ori_type=None
        if target_node.data_types[0] is not  None:
            ori_type=" ".join(target_node.data_types[0])
        new_decl=self.getNewDecl(ori_type,target_node_name)
        new_attackDict=[]
        for pos in target_node.can_new_decl_add_pose:
            _attackDict = copy.deepcopy(self.attackDict)
            is_succ=InsAattack(_attackDict,pos,new_decl)
            if is_succ:
                new_attackDict.append(_attackDict)
        return new_attackDict,target_node_name
    # stage2==========addOldNode=============
    # stage2==========addNewNode=============
    def addNewNode(self,n_candidates=5,choose=1):

        if choose==1:
            if self.high_frequen_tokens is None:
                self.high_frequen_tokens=self.get_valid_tokens_name(self.config['topCommonTokens'])         
            while True:
                choose_name=random.choice(self.high_frequen_tokens)
                if choose_name not in self.codeinfo.valid_variable_names:
                    break
        else:

            max_node_name_len=max([len(var) for var in self.codeinfo.valid_variable_names ])
            while True:
                new_node_name_len=random.choice(range(1,max_node_name_len))
                choose_name=''.join(random.sample(self.letters,new_node_name_len))
                if choose_name not in self.codeinfo.valid_variable_names:
                    break
        new_decl=self.getNewDecl(None,choose_name)

        can_add_lines=[-1]+self.codeinfo.StmtInsPos_line
        n_candidates=min(len(can_add_lines),n_candidates)
        choose_lines=random.sample(can_add_lines,n_candidates)
        new_attack_dicts=[]
        for add_pos in choose_lines:
            if add_pos !=-1:
                add_pos=self.codeinfo.tokenStmts2tokensPos[add_pos,len(self.codeinfo.token_stmts[add_pos])-1]
            _attackDict = copy.deepcopy(self.attackDict)
            is_succ=InsAattack(_attackDict,add_pos,new_decl)
            if is_succ:
                new_attack_dicts.append(_attackDict)
        return new_attack_dicts,choose_name
    # stage2==========addNewNode=============
    # stage2==========addSubGraph=============
    def update_body_choose(self,pos):
        if self.codeinfo.pose_and_body_list.get(pos) is not None:
            return self.codeinfo.pose_and_body_list[pos]
        body_list=[]
        pos_index=self.codeinfo.StmtInsPos_token.index(pos)
        cur_prefix=self.codeinfo.stmtIns_pos_prefix[pos_index]
        cur_line=self.codeinfo.tokensPos2tokenStmts[pos][0]
        for line in range(cur_line,-1,-1):
            if line not in self.codeinfo.one_line_sent:
                continue 
            if self.codeinfo.token_stmts[line][0] in type_words:
                body_list.append(line)
                continue
            if self.codeinfo.token_stmts[line][0] in ['strcmp']:
                continue
            line_prefix=self.codeinfo.stmts_line_prefix[line]
            if self.cheack_prefixs(line_prefix,cur_prefix):
                body_list.append(line)
        for block_name,block in self.codeinfo.valid_blocks.items():
            if block_name.split('_')[0] not in ['if','for','while','dowhile']:
                    continue
            if block.end_pos< pos:
                body_list.append(block_name)
            else:
                can_add=True
                for i,var in  enumerate(block.using_var):
                    if var in self.codeinfo.func_blocks.keys():
                        if pos>self.codeinfo.func_blocks[var].func_name_pos[0]:
                            if self.codeinfo.func_blocks[var].decl_prefix is None or self.cheack_prefixs(self.codeinfo.func_blocks[var].decl_prefix,cur_prefix):
                                continue 
                    elif var in self.codeinfo.data_blocks.keys():
                        if self.codeinfo.data_blocks[var].pos[0]<pos:
                                continue 

                    elif var in self.codeinfo.valid_nodes.keys():
                        satisfy_prefix=False
                        for j,prefix in enumerate(self.codeinfo.valid_nodes[var].prefixs):
                            if self.codeinfo.valid_nodes[var].poses[j]<pos:
                                if self.cheack_prefixs(prefix,cur_prefix):
                                    satisfy_prefix=True
                                    break
                            else:
                                break
                        if satisfy_prefix:continue            
                    can_add=False
                    break        
                if can_add:
                    body_list.append(block_name)
        self.codeinfo.pose_and_body_list[pos]=body_list
        return body_list
    
    def addSubGraph(self,n_candidates=5):
       
        limitation=self.change_limits-self.attack_changes_nums
        condition_head=random.choice(self.condition_head)
        n_candidates+=self.gen_bound
        cond_candition=random.randint(1,n_candidates//2)
        body_candition=n_candidates//cond_candition+1
        choose_conditions_and_pos=self.gen_subGraph_condition(cond_candition)
        new_attack_dicts=[]
        for condition,add_pos in choose_conditions_and_pos:
            if condition == []:continue
            if len(new_attack_dicts)>n_candidates:
                break
            sub_code=(condition_head.replace('condition',condition)).split()
            body_limt=limitation-len(sub_code)-2
            if body_limt<0:continue
            cand_body_list=['random_print','fixed_code']+self.update_body_choose(add_pos)
            bodys=self.gen_subGraph_body(cand_body_list,body_candition,body_limt)
            if bodys==[]:bodys.append([';'])
            for body in bodys:
                _attackDict = copy.deepcopy(self.attackDict)
                is_succ=InsAattack(_attackDict,add_pos,sub_code+['{']+body+['}'])
                if is_succ:
                    new_attack_dicts.append(_attackDict)
            
                    
        return new_attack_dicts,len(new_attack_dicts)*[None]
    
    def gen_subGraph_condition(self,choose=0,cond_candidates=5):
        """生成条件，choose:0是node相关，choose:1是random"""
        choose_conditions_and_pos=[]
        choose=random.choice([0,1])
        if choose==0:
            candi_conditions_and_pos=self.get_node_condition_n_targets()
            
            if len(candi_conditions_and_pos)>0:
               
                choose_conditions_and_pos=random.sample(candi_conditions_and_pos,min(cond_candidates,len(candi_conditions_and_pos)))

        if choose==1 or len(choose_conditions_and_pos)<cond_candidates :
            for i in range(cond_candidates-len(choose_conditions_and_pos)):
                choose_condition=self.get_random_condition()
                add_pos=random.choice(self.codeinfo.StmtInsPos_token)
                choose_conditions_and_pos.append((choose_condition,add_pos))
        return choose_conditions_and_pos

    def gen_subGraph_body(self,body_list,body_candidates,limitation=100):
        
        choose_body_types=random.sample(body_list,len(body_list))
        gen_bodys=[]
        
        for body_choose in choose_body_types:            
            if body_choose=='random_print':
                body_cand = ["printf ( \"%s\" ) ;"]
                msg = ['err','crash','alert','warning','flag','exception','level','create','delete','success','get','set',''.join(random.choice(self.letters) for i in range(4))]
                cand_body=(random.choice(body_cand)%(random.choice(msg))).split(" ")
            elif body_choose=='fixed_code':
                cand_body=random.choice(self.fixed_inserts).split()
            elif isinstance(body_choose,int):
                cand_body=self.codeinfo.token_stmts[body_choose]
            else:
                target_block=self.codeinfo.valid_blocks[body_choose]
                cand_body=self.codeinfo.code_tokens[target_block.begin_pos:target_block.end_pos+1]
            if len(cand_body)<limitation:
                gen_bodys.append(cand_body)
            if len(gen_bodys)>body_candidates:
                break
                
        return gen_bodys

    def get_node_condition_n_targets(self,n_candidates=5):
        if self.attack_infos['addSubGraph']['targets'] in [None,[]]:
            return None,None
       
        n_candidates=min(n_candidates,len(self.attack_infos["addSubGraph"]['targets']))
        n_candidates=len(self.attack_infos["addSubGraph"]['targets'])
        choose_node_names=random.sample(self.attack_infos["addSubGraph"]['targets'],n_candidates)
        
        
        candi_node_and_conditions=[]
        for node_name in choose_node_names:
            cand_condition_and_pose,_=self.get_node_condition(node_name)
            if cand_condition_and_pose is None:
                self.attack_infos['addSubGraph']['targets'].remove(node_name)
                continue
            candi_node_and_conditions+=cand_condition_and_pose
    
        return candi_node_and_conditions
    
    def get_node_condition(self,target_node_name=None):
        
        if target_node_name is None:
          
            target_node_name=random.choice(self.attack_infos["addSubGraph"]['targets'])
        target_node=self.codeinfo.valid_nodes[target_node_name]
        if target_node.can_gen_condition_value_pos_and_can_add_pos==None:
            return None,target_node_name
        if target_node.cand_condition_and_pose !=[]:
            return target_node.cand_condition_and_pose,target_node_name
        if target_node.can_gen_condition_value_pos_and_can_add_pos=={}:
            for i,value in enumerate(target_node.values):
                if value is None or value in ['rvalue']:
                    continue
                target_node.can_gen_condition_value_pos_and_can_add_pos[i]={'condition':[],'canAddpos':[]}
            if target_node.can_gen_condition_value_pos_and_can_add_pos=={}:
                target_node.can_gen_condition_value_pos_and_can_add_pos=None
                return None,target_node_name
            can_gen_condition_pos=list(target_node.can_gen_condition_value_pos_and_can_add_pos.keys()) 
            for i,pos_index in enumerate(can_gen_condition_pos):
                begin_pos_indx=pos_index
                if i!=len(can_gen_condition_pos)-1:
                    
                    end_pos_indx=can_gen_condition_pos[i+1]
                else:end_pos_indx=-1
                can_add_pos=self.get_node_condition_at_valuse_pos(target_node,begin_pos_indx,end_pos_indx)
                if can_add_pos!=[]:
                    target_node.can_gen_condition_value_pos_and_can_add_pos[pos_index]['canAddpos']= can_add_pos
                    
                    cur_val=target_node.values[pos_index]
                    whole_name=target_node.whole_names[pos_index]
                    if '[ ]' in whole_name or '[]' in whole_name or 'return' in whole_name:
                        
                        continue        
                    if cur_val in ['input','UnaryChanged','func init'] or '{' in cur_val:
                        condition=whole_name+' ==  NULL' 
                    else:
                        condition=whole_name+' != '+ cur_val
                    target_node.can_gen_condition_value_pos_and_can_add_pos[pos_index]['condition']=condition
                else:
                    del target_node.can_gen_condition_value_pos_and_can_add_pos[pos_index]
       
        cand_condition_and_pose=[]
        for val_pos in target_node.can_gen_condition_value_pos_and_can_add_pos.keys():
            condition=target_node.can_gen_condition_value_pos_and_can_add_pos[val_pos]['condition']
            for can_add_pos in target_node.can_gen_condition_value_pos_and_can_add_pos[val_pos]['canAddpos']:
                cand_condition_and_pose.append((condition,can_add_pos))
        target_node.cand_condition_and_pose=cand_condition_and_pose
        return cand_condition_and_pose,target_node_name
    
    def get_node_condition_at_valuse_pos(self,target_node,begin_pos_indx,end_pos_indx):
        begin_pos=target_node.poses[begin_pos_indx]
        if end_pos_indx==1: end_pos=len(self.codeinfo.code_tokens)-1
        end_pos=target_node.poses[end_pos_indx]
        cur_prefix=target_node.prefixs[begin_pos_indx]
        can_add_pos=[]
        for pos in range(begin_pos+1,end_pos):
            if pos in  self.codeinfo.StmtInsPos_token:
                pos_index=self.codeinfo.StmtInsPos_token.index(pos)
                pos_prefix=self.codeinfo.stmtIns_pos_prefix[pos_index]
                if self.cheack_prefixs(cur_prefix,pos_prefix):
                    can_add_pos.append(pos)  
        return can_add_pos
   
    def get_random_condition(self,seed=2022):
        
        rand_maxnum=random.randint(0,seed)
        condition=""
        funs = {	
                'sin': [-1,1], 
                'cos': [-1,1], 
                'exp': [1,3],  
                'sqrt': [0,1], 
                'rand': [0,rand_maxnum]  
                }
        func = random.choice(list(funs.keys()))
        condition += func + " ( "
        if func == "rand":
            condition += " ) "+" % "+str(rand_maxnum)+" "
        else:
            condition += "%.2f ) "%random.random()
        ops = [' <  ', ' > ', " <= ", " >= ", " == "]
        op = random.choice(ops)
        condition += op 
        if op in [" < "," <= "," == "]:
            condition += str(int(funs[func][0] - 100*random.random()))
        else:
            condition += str(int(funs[func][1] + 100*random.random()))
        
        return condition
    # stage1==========addSubGraph=============
    # stage1==========recons var node=============
    def moveNode_n_targets(self,n_candidates=5):
        
        choose_node_names=random.sample(self.attack_infos['moveNode']['targets'],len(self.attack_infos['moveNode']['targets']))
        new_codes_tokens=[]
        gen_nums=0
        for node_name in choose_node_names:
            
            code_tokens,_=self.moveNode(node_name)
            new_codes_tokens.append(code_tokens)
            gen_nums+=len(code_tokens)
            if gen_nums>n_candidates:
                break
        return new_codes_tokens,choose_node_names
    def moveNode(self,target_node_name=None):

        if target_node_name is None:
            target_node_name=random.choice(self.attack_infos['moveNode']['targets'])
        target_node =self.codeinfo.valid_nodes[target_node_name]
        valid_func_prefix=[key for key in target_node.groups.keys() if key is None or key.startswith('func')]

        node_candidate_tokens=[]
        if len(valid_func_prefix)>0:
            for target_func in valid_func_prefix:
                func_code_tokens,_=self.moveNode_target_node_funcs(target_node,target_func)
                if func_code_tokens is not None:
                    for code_tokens in func_code_tokens:
                        node_candidate_tokens.append(code_tokens)
        else:
            func_code_tokens,_=self.moveNode_target_node_funcs(target_node,None)# 
            if func_code_tokens is not None:
                    for code_tokens in func_code_tokens:
                        node_candidate_tokens.append(code_tokens)

        return node_candidate_tokens,target_node_name
    def moveNode_target_node_funcs(self,target_node,target_func='random'):
        
        if target_func=='random':
        
            valid_func_prefix=[key for key in target_node.groups.keys() if key is None or key.startswith('func')]
            if len(valid_func_prefix)>0:
                target_func=random.choice(valid_func_prefix)
            else:
                target_func=None
        try:
            func_value_poses= target_node.groups[target_func]

            pos_begin_indx=target_node.poses.index(func_value_poses[0])
            if target_func is None:
                pos_end_indx=len(target_node.poses)-1
            else:
                pos_end_indx=target_node.poses.index(func_value_poses[-1])
        except:
            pos_begin_indx=0
            pos_end_indx=len(target_node.poses)-1
        can_add=False
        include_None=True
        if target_node.groups.get(None) is  None and target_func is not None:
            can_add_lines=[-1]
        else:
            include_None=False
            can_add_lines=[]
       
        for i,(pos,val) in enumerate(zip(target_node.poses[pos_begin_indx:pos_end_indx+1],target_node.values[pos_begin_indx:pos_end_indx+1])):
            if val =='func init':break
            if val is not None and val  not in ['rvalue','selfOp']:
                if self.codeinfo.tokensPos2tokenStmts[pos][0]!=0:
                    whole_name=target_node.whole_names[pos_begin_indx+i]
                    if '.' not in whole_name and '[' not in whole_name and'(' not in whole_name:
                        can_add=True
                        break
                    elif '.' in whole_name or '[' in whole_name:break 
        if not can_add:
            return None,target_func
        curr_prefix=target_node.prefixs[pos_begin_indx+i]
        node_type=target_node.data_types[pos_begin_indx+i]
        if node_type is None:return None,target_func
        if " ".join(node_type) in self.codeinfo.typedef_node.keys():
            after_pos=self.codeinfo.valid_nodes[" ".join(node_type)].poses[0]
            after_line=self.codeinfo.tokensPos2tokenStmts[after_pos][0]
            if -1 in  can_add_lines and after_line>0:
                can_add_lines.remove(-1)
        else:
            after_line=0
        whole_name=target_node.whole_names[pos_begin_indx+i]
        cur_line,line_pos=self.codeinfo.tokensPos2tokenStmts[pos]
        cur_content=self.codeinfo.token_stmts[cur_line]
        
        decl_add_content=None
        changed_decl=[]
        before_decl_line=None
        before_changed_decl=[]
        in_for_condition=False
        if i==0 and include_None: 
            before_prefix=curr_prefix
            if 'for' in cur_content:
                for_pos=go4before(cur_content,'for',line_pos)
                if for_pos!=-1 and (line_pos-for_pos)<5:
                    in_for_condition=True
            if cur_content.count('for')>1:
                return None,target_func
           
            decl_add_content=(" ".join(node_type)+" "+whole_name+ " ; ").split()
            if val.startswith("'") or val.startswith('"'):
                val=[val]
            else:
                val=val.split()
            init_content=whole_name.split()+['=']+val
            
            end_pos=go4next(cur_content,';',line_pos)
            if go4before(cur_content,'(',line_pos)!=-1:
                begin_pos=go4before(cur_content,'(',line_pos)+1
            else: begin_pos=0
           

            line_vars_nums=self.codeinfo.get_line_decl_var(cur_content,begin_pos,end_pos)
            if in_for_condition:
                if line_vars_nums==1:
                    changed_decl=self.remove_var_from_decl(cur_line,node_type,is_in_for=in_for_condition,begin_pos=for_pos,vars=line_vars_nums)
                else:
                    changed_decl=init_content+[';']+self.remove_var_from_decl(cur_line,init_content,in_for_condition,for_pos,line_vars_nums)
            else:
                if line_vars_nums==1 or len(cur_content)==len(init_content)+len(node_type)+1:
                    changed_decl=init_content+[';']
                else:
                
                    if random.random()<0.5:
                        changed_decl=self.remove_var_from_decl(cur_line,init_content)+init_content+[';']
                    else:
                        changed_decl=init_content+[';']+self.remove_var_from_decl(cur_line,init_content)
        elif i==0 and not include_None:
            before_prefix=curr_prefix
           
            if target_node.groups.get(None) is not None and target_node.groups[None][0]<pos:
                before_whole_name_index=target_node.poses.index(target_node.groups[None][0])
                before_whole_name=target_node.whole_names[before_whole_name_index]
            else:
                before_whole_name=whole_name

            decl_add_content=(" ".join(node_type)+" "+before_whole_name+ " ; ").split()
            changed_decl=cur_content
        elif i>0:
            changed_decl=cur_content
            j=i-1
            is_find=False
            while j:
                before_whole_name=target_node.whole_names[pos_begin_indx+j]
                if '.' not in before_whole_name and '[' not in before_whole_name and'(' not in before_whole_name:
                    if target_node.values[pos_begin_indx+j] not in ['rvalue','selfOp']:
                        befor_pos=target_node.poses[pos_begin_indx+j]
                        befor_line,befor_line_pos=self.codeinfo.tokensPos2tokenStmts[befor_pos]
                        befor_line=self.codeinfo.token_stmts[befor_line]
                        if "for" in befor_line and '(' in  befor_line[befor_line.index('for'):befor_line_pos]:
                            pass
                        elif "if" in befor_line:
                            pass
                        else:
                            is_find=True
                            break
                j-=1

            if not is_find:    
                return None,target_func

            before_prefix=target_node.prefixs[pos_begin_indx+j]
            before_whole_name=target_node.whole_names[pos_begin_indx+j]
            
            if self.cheack_prefixs(before_prefix,curr_prefix):
              
                decl_add_content=(" ".join(node_type)+" "+before_whole_name+ " ; ").split()
                
                before_decl_line=self.codeinfo.tokensPos2tokenStmts[target_node.poses[pos_begin_indx+i-1]][0]
                before_decl=self.codeinfo.token_stmts[before_decl_line]
                if before_decl==decl_add_content: 
                    
                    before_changed_decl=[]
                else:
                    before_changed_decl=self.remove_var_from_decl(before_decl_line,before_whole_name.split())
            else:
                return None,target_func

        for line in range(cur_line-1,after_line-1,-1):
            if line in self.codeinfo.StmtInsPos_line and self.cheack_prefixs(self.codeinfo.stmts_line_prefix[line],before_prefix,include_None):
                can_add_lines.append(line)
        if before_decl_line is not None and before_changed_decl==[]:
            if (before_decl_line-1) in can_add_lines:
                can_add_lines.remove(before_decl_line-1)
            if (before_decl_line) in can_add_lines:
                can_add_lines.remove(before_decl_line)
       
        if len(can_add_lines)==0:
            
            return None,target_func
        else: 
            
            new_code_tokens=[]
            for add_line in can_add_lines:
               
                new_tokens=[]
                if add_line==-1:
                    new_tokens+=decl_add_content
                for line in range(len(self.codeinfo.token_stmts)):
                    if line==cur_line:
                        new_tokens+=changed_decl
                    elif line==before_decl_line:
                        new_tokens+=before_changed_decl
                    else:
                        new_tokens+=self.codeinfo.token_stmts[line]
                    if line == add_line: 
                        new_tokens+=decl_add_content
                new_code_tokens.append(new_tokens)
            return new_code_tokens,target_func
    # stage1==========rrecons var node=============
    # stage1==========recons data block=============
    def modifyDatablock_n_targets(self,n_candidates=5):
        choose_target_names=random.sample(self.attack_infos['modifyDatablock']['targets'],len(self.attack_infos['modifyDatablock']['targets']))
        new_codes_tokens=[]
        gen_nums=0
        for target_name in choose_target_names:
            code_tokens,_ = self.modifyDatablock(target_name)
            new_codes_tokens.append(code_tokens)
            gen_nums+=len(code_tokens)
            if gen_nums>n_candidates:
                break
        return new_codes_tokens,choose_target_names   
    def modifyDatablock(self,target_block_name=None):
        
        if target_block_name is None:
            target_block_name=random.choice(self.attack_infos['modifyDatablock']['targets'])
        target_block=self.codeinfo.data_blocks[target_block_name]
        
        
        if target_block.first_using_pos is None:
           
            prefix=[]
            target_pos_end_line=len(self.codeinfo.token_stmts)-1
        elif target_block.decl_pos!=target_block.def_poses[0]: 
            prefix=target_block.outer_prefix
            target_pos_end_line=len(self.codeinfo.token_stmts)-1
        else:
            prefix=target_block.outer_prefix
            target_pos_end_line=self.codeinfo.tokensPos2tokenStmts[target_block.first_using_pos][0]-1
        
        block_decl_begin_line=self.codeinfo.tokensPos2tokenStmts[target_block.def_poses[0]][0]
        block_decl_end_line=self.codeinfo.tokensPos2tokenStmts[target_block.def_poses[1]][0]
        if target_block.after_line==-1:
            can_add_lines=[-1]
        else:
            can_add_lines=[]
        for line in range(target_pos_end_line,target_block.after_line,-1):
            if line in self.codeinfo.StmtInsPos_line:
                
                line_prefix=self.codeinfo.stmts_line_prefix[line]
               
                if self.cheack_prefixs(line_prefix,prefix) or self.codeinfo.StmtIndents[line]==0:
                    can_add_lines.append(line)
        can_add_lines=[line for line in can_add_lines if  line!=block_decl_begin_line-1 and line!=block_decl_end_line]
      
        candidate_tokens=[]
        if len(can_add_lines)>0:

           
            add_line_def=list(range(block_decl_begin_line,block_decl_end_line+1))
            add_line_using=[]
            attention_lines=[]
            if len(target_block.pos)>1 and target_block.def_poses[0]==target_block.decl_pos:
                for pos in target_block.pos[1:]:
                    line,_=self.codeinfo.tokensPos2tokenStmts[pos]# 
                   
                    if '=' in self.codeinfo.token_stmts[line] or '(' in self.codeinfo.token_stmts[line]:
                        attention_lines.append(line)
                    elif line not in add_line_using and line not in add_line_def:
                        add_line_using.append(line)
                    if line>=target_pos_end_line:
                        break
            
            if len(attention_lines)>0:
                can_add_lines=[line for line in can_add_lines if line<min(attention_lines)]
            can_add_lines.sort()
            for i in range(len(can_add_lines)):
                choose_line1=can_add_lines[i]
                if add_line_using!=[]:
                    choose_last=len(can_add_lines)
                else:
                    choose_last=i+1
                for j in range(i,choose_last):
                    choose_line2=can_add_lines[j]
                    
                    new_code_tokens=[]
                    for i in range(-1,len(self.codeinfo.token_stmts)):
                        if i not in (add_line_def+add_line_using) and i!=-1:
                            new_code_tokens+=self.codeinfo.token_stmts[i]
                        if i ==choose_line1:
                            for line in add_line_def:
                                new_code_tokens+=self.codeinfo.token_stmts[line]
                                if line==add_line_def[-1] and self.codeinfo.token_stmts[line][-1]!=';':
                                    new_code_tokens+=[';']
                        if i == choose_line2:
                            for line in add_line_using:
                                new_code_tokens+=self.codeinfo.token_stmts[line]
                    candidate_tokens.append(copy.deepcopy(new_code_tokens))
        else:
            return candidate_tokens,target_block_name
      
        return candidate_tokens,target_block_name
    # stage1==========recons data block=============
    # stage1==========recons func block=============
    def modifyFuncBlock_n_targets(self,n_candidates=5):
        
        n_candidates=min(n_candidates,len(self.attack_infos['modifyFuncblock']['targets']))
        choose_target_names=random.sample(self.attack_infos['modifyFuncblock']['targets'],n_candidates)
        new_codes_tokens=[]
        gen_nums=0
        for target_name in choose_target_names:
            code_tokens,_ = self.modifyFuncBlock(target_name)
            if code_tokens is not None:
                new_codes_tokens.append(code_tokens)# 返回得是tokens得list
                gen_nums+=len(code_tokens)
            if gen_nums>n_candidates:
                break
        return new_codes_tokens,choose_target_names
    def modifyFuncBlock(self,target_func_name=None):
       
        if  target_func_name is None:
            target_func_name=random.choice(self.attack_infos['modifyFuncblock']['targets'])
        target_func=self.codeinfo.func_blocks[target_func_name]
        
        try:
            
            first_call_pos_line=self.codeinfo.tokensPos2tokenStmts[target_func.call_poses[0]][0]
        except:
            first_call_pos_line=len(self.codeinfo.token_stmts)
        prefix=target_func.func_call_prefix
        
        old_decl_line=self.codeinfo.tokensPos2tokenStmts[target_func.func_name_pos[0]][0]
        
        if target_func.def_poses==[]:
            old_def_begin_line=old_def_end_line=-1
        else:
            old_def_begin_line=self.codeinfo.tokensPos2tokenStmts[target_func.def_poses[0][0]][0]
            old_def_end_line=self.codeinfo.tokensPos2tokenStmts[target_func.def_poses[0][1]][0]
        
        if old_decl_line==old_def_begin_line:
            func_decl=self.codeinfo.token_stmts[old_decl_line][:-1]+[';']
            if func_decl[0]  in [ target_func_name,'*']:
                func_decl=['void']+func_decl
        else:
            if target_func.whole_name.startswith('*'):
                if old_def_begin_line!=-1:
                    func_type_pos=self.codeinfo.token_stmts[old_def_begin_line].index('*')
                    func_type=self.codeinfo.token_stmts[old_def_begin_line][:func_type_pos]
                else:
                    func_type_pos=self.codeinfo.token_stmts[old_decl_line].index('*')
                    func_type=self.codeinfo.token_stmts[old_decl_line][:func_type_pos]
                func_decl_begin_pos=self.codeinfo.token_stmts[old_decl_line].index(target_func_name.lstrip('*'))-target_func.whole_name.count('*')

            else:
                if old_def_begin_line!=-1:
                    func_type_pos=self.codeinfo.token_stmts[old_def_begin_line].index(target_func_name)
                    func_type=self.codeinfo.token_stmts[old_def_begin_line][:func_type_pos]
                else:
                    func_type_pos=self.codeinfo.token_stmts[old_decl_line].index(target_func_name)
                    func_type=self.codeinfo.token_stmts[old_decl_line][:func_type_pos]
                func_decl_begin_pos=self.codeinfo.token_stmts[old_decl_line].index(target_func_name)
            
            if func_type ==[]:func_type=['void']
            func_decl_end_pos=go4match(self.codeinfo.token_stmts[old_decl_line],'(',func_decl_begin_pos)
            func_decl_without_type=self.codeinfo.token_stmts[old_decl_line][func_decl_begin_pos:func_decl_end_pos+1]
            func_decl=func_type+func_decl_without_type+[';']
        after_line=target_func.after_line
       
        can_add_decl_lines=[-1]
        can_add_def_lines=[-1,len(self.codeinfo.token_stmts)-1]
        for line in range(first_call_pos_line,after_line,-1):
            if line in self.codeinfo.StmtInsPos_line:
               
                line_prefix=self.codeinfo.stmts_line_prefix[line]
                
                if self.cheack_prefixs(line_prefix,prefix) or self.codeinfo.StmtIndents[line]==0:
                    can_add_decl_lines.append(line)
        can_add_decl_lines=[line for line in can_add_decl_lines if line>=after_line]
        for line in range(len(self.codeinfo.token_stmts)):
            if self.codeinfo.StmtIndents[line]==0:
                if not len(set(self.codeinfo.token_stmts[line])&set(target_func.def_vars))>0:
                    
                    can_add_def_lines.append(line)
                else:
                    can_add_def_lines=[l for l in can_add_def_lines if l > line]
                        
        can_add_def_lines=[line for line in can_add_def_lines if  line >target_func.def_after_line]
         
        candidate_tokens=[]
        if old_def_end_line==-1:
            def_lines=[]
        else:
            def_lines=list(range(old_def_begin_line,old_def_end_line+1))
        if len(can_add_decl_lines)>0:
            if self.codeinfo.token_stmts[old_decl_line]!=func_decl and old_decl_line!=old_def_begin_line :
                try:
                    old_decl=self.remove_var_from_decl(old_decl_line,func_decl_without_type)
                except:
                    func_decl_begin_pos=func_decl.index(target_func_name)
                    block_end_pos=go4match(func_decl,'(',func_decl_begin_pos)
                    old_decl=self.remove_var_from_decl(old_decl_line,func_decl[func_decl_begin_pos:block_end_pos+1])
                
            else:old_decl=[]
            
            for choose_decl_line in can_add_decl_lines:
                
                new_can_add_def_lines=[line for line in can_add_def_lines if line>=choose_decl_line]
                for choose_def_line in new_can_add_def_lines:
                    if choose_def_line not in [old_decl_line-1,old_decl_line] or choose_def_line not in [old_def_begin_line-1,old_def_end_line]:
                        new_code_tokens=[]
                        for i in range(-1,len(self.codeinfo.token_stmts)):
                            if i not in (def_lines+[old_decl_line]) and i!=-1:
                                new_code_tokens+=self.codeinfo.token_stmts[i]
                            if i==old_decl_line:
                                new_code_tokens+=old_decl
                            if i ==choose_decl_line:
                                new_code_tokens+=func_decl
                            if i == choose_def_line:
                                for line in def_lines:
                                    new_code_tokens+=self.codeinfo.token_stmts[line]
                        candidate_tokens.append(copy.deepcopy(new_code_tokens))
            return candidate_tokens,target_func_name
        else:
            return None,target_func_name
    # stage1==========recons func block=============
    # stage1==========recons for while etc, block=============
    def modifyblock_n_targets(self,n_candidates=5):

        choose_block_names=random.sample(self.attack_infos['ReconsBlock']['targets'],len(self.attack_infos['ReconsBlock']['targets']))# 打乱顺序
        new_codes_tokens=[]
        gen_nums=0
        for target_block_name in choose_block_names:
            code_tokens,_=self.modifyblock(target_block_name)
            new_codes_tokens.append([code_tokens])
            gen_nums+=1
            if gen_nums>n_candidates:
                break
        return new_codes_tokens,choose_block_names
    def modifyblock(self,target_block_name=None):

        if target_block_name is None:
            target_block_name=random.choice(self.attack_infos['ReconsBlock']['targets'])
        target_block=self.codeinfo.valid_blocks[target_block_name]

        block_type,_=target_block_name.split('_')
        new_code_tokens=None
        if block_type=='if':
            new_code_tokens=self.reconstructIf(target_block)
        elif block_type=='for':
            new_code_tokens=self.reconstructFor(target_block)
        elif block_type=='while':
            new_code_tokens=self.reconstructWhile(target_block)
        elif block_type=='dowhile':
            new_code_tokens=self.reconstructDoWhile(target_block)
        
        return new_code_tokens,target_block_name
    
    def reconstructIf(self,block:valid_block):

        begin_pos=block.last_if_pos
        tokens=self.codeinfo.code_tokens
        
        ifIdx = begin_pos
        conditionEndIdx = go4match(tokens, "(", ifIdx)
        
        end_pos=self.codeinfo.tokensEnd[begin_pos]
        
        single_if=True
        if block.last_if_pos_has_else:
            single_if=False
                
        if single_if:
            ifBlockEndIdx = end_pos
            pos=[ifIdx, conditionEndIdx, ifBlockEndIdx]
            new_code_tokens=IfReplace(self.codeinfo.code_tokens,pos)

        else:
            if tokens[conditionEndIdx + 1] == "{":
                ifBlockEndIdx = go4match(tokens, "{", conditionEndIdx + 1)
            elif tokens[conditionEndIdx + 1] in ['for','if','do','while']:

                ifBlockEndIdx=self.codeinfo.tokensEnd[conditionEndIdx+1]
            else:
                ifBlockEndIdx = go4next(tokens, ";", conditionEndIdx + 1)
            
            elseBlockEndIdx = end_pos
            pos=[ifIdx, conditionEndIdx, ifBlockEndIdx, elseBlockEndIdx]
            new_code_tokens=IfElseReplace(tokens,pos)
        return new_code_tokens
    def reconstructFor(self,block:valid_block):

        tokens=self.codeinfo.code_tokens
        forIdx = block.begin_pos
        conditionEndIdx = go4match(tokens, "(", forIdx)
        if tokens[conditionEndIdx + 1] == "{":
            blockForEndIdx = go4match(tokens, "{", conditionEndIdx)
        else:
            blockForEndIdx = block.end_pos
        condition1EndIdx = go4next(tokens, ";", forIdx)
        condition2EndIdx = go4next(tokens, ";", condition1EndIdx + 1)
        pos=[forIdx, condition1EndIdx, condition2EndIdx, conditionEndIdx, blockForEndIdx]
        new_code_tokens=For2WhileRepalce(tokens,pos)
        return new_code_tokens
    def reconstructWhile(self,block:valid_block):
        tokens=self.codeinfo.code_tokens
        whileIdx = block.begin_pos
        conditionEndIdx = go4match(tokens, "(", whileIdx)
        if tokens[conditionEndIdx + 1] == "{":
            blockWhileEndIdx = go4match(tokens, "{", conditionEndIdx)
        else:
            blockWhileEndIdx = block.end_pos
        pos=[whileIdx, conditionEndIdx, blockWhileEndIdx]
        new_code_tokens=While2ForRepalce(tokens,pos)
        return new_code_tokens
    def reconstructDoWhile(self,block:valid_block):
        tokens=self.codeinfo.code_tokens
        doIdx=block.begin_pos
        if tokens[doIdx+1]=="{":
            blockDoEndIdx = go4match(tokens, "{", doIdx)
        else:
            blockDoEndIdx= go4next(tokens,'while',doIdx)-1
        whileIdx=blockDoEndIdx+1
        conditionEndIdx= go4match(tokens, "(", whileIdx)
        pos=[doIdx,blockDoEndIdx,whileIdx,conditionEndIdx]
        new_code_tokens=DoWhile2ForRepalace(tokens,pos)
        return new_code_tokens
    # stage1==========recons for while etc, block=============    
    ##=====tools function=========================
    def remove_var_from_decl(self,line:int,var:list,is_in_for=False,begin_pos=0,vars=1):
        
        tokens_line=self.codeinfo.token_stmts[line]
        
        match=False
        width=len(var)
        for pos,t in enumerate(tokens_line[begin_pos:]):
            if t == var[0]:
                match=True
                for i in range(1,width):
                    if tokens_line[pos+i]!=var[i]:
                        match=False
                        break
            if match:
                break
        line_pos=begin_pos+pos
        
        new_tokens_line=[]
        if is_in_for and vars==1:
                new_tokens_line=tokens_line[:line_pos]+tokens_line[line_pos+width:]
        else:
            if tokens_line[line_pos-1]==',': 
                new_tokens_line=tokens_line[:line_pos-1]+tokens_line[line_pos+width:]
            else:
                new_tokens_line=tokens_line[:line_pos]+tokens_line[line_pos+width+1:]
        
        return new_tokens_line
    @ staticmethod
    def generateNodeRedundant(node_nums=None,bounds=5):
        candiEqualSelf=['node','node & node','node | node']
        candiEqualOne=['(  node / node ) ',' 1 ']
        candiEqualZero=['( node - node ) ',' 0 ']
        if node_nums==None:
            node_nums=random.randint(4,20)
        cur_node_nums=2
        expression=['node','=']
        
        random_self=random.choice(candiEqualSelf)
        expression+=(random_self.split())
        cur_node_nums+=len(random_self.split())
        while cur_node_nums<=node_nums-1:
            choose=random.randint(0,2)
            if choose==0:
                op=random.choice(['+','-']).split()
                add_exp=op+random.choice(candiEqualZero).split()
            elif choose==1:
                op=random.choice(['*','|']).split()
                add_exp=op+random.choice(candiEqualOne).split()  
            elif choose==2:
                op=random.choice([' & ',' | ']).split()
                add_exp=op+random.choice(candiEqualSelf).split()
            add_nums=len(add_exp)
            if add_nums<= node_nums+bounds-cur_node_nums-1:
                expression+=add_exp
                cur_node_nums+=add_nums
            if node_nums-cur_node_nums-1-add_nums <bounds:

                break
        
        return " ".join(expression+[';'])
    @ staticmethod
    def getNewDecl(ori_type=None,target_node_name=None):

        if ori_type is not None:
            new_type=ori_type
            while new_type==ori_type:
                new_type=random.choice(type_words)
            return [new_type,target_node_name,';']
        else:
            new_type=random.choice(type_words)
            return [new_type,target_node_name,';']
    @staticmethod
    def strict_cheack_prefixs(prefix_before,prefix_after):

        if prefix_before!=prefix_after:
            return False
        else:
            return True
    @staticmethod
    def cheack_prefixs(prefix_before,prefix_after,include_None=True):
        if prefix_after=='any':
            return True
        if prefix_after is None or prefix_after==[]:
            if prefix_before!=prefix_after:
                return False

        try: 
            if len(prefix_before)==0:
                if include_None:
                    return True
                else:
                    return False 
        except:
            if prefix_before==None:
                if include_None:
                    return True
                else:
                    return False  
        try:
            compare_prefix=copy.deepcopy(prefix_before)
            if compare_prefix[-1].startswith('scanf'):
                compare_prefix.pop(-1)
            if len(compare_prefix)<=len(prefix_after) and compare_prefix==prefix_after[:len(compare_prefix)]:
                return True
            else:
                return False
        except:
            print("something wrong in prefix cheacking")
    @staticmethod
    def get_out_prefix(prefix1:list,prefix2:list):

        if prefix1 is None: return []
        if prefix2 is None:return []
        min_prefix_len=min(len(prefix1),len(prefix2))
        outer_prefix=[]
        for i in range(min_prefix_len):
            if prefix1[i]==prefix2[i]:
                outer_prefix.append(prefix1[i])
        return outer_prefix
    @staticmethod
    def check_selfOp(whole_name,assin):

        whole_name_tokens=whole_name.split()
        assin_tokens=assin.split()
        is_self_op=False
        for token in whole_name_tokens:
            if token not in ops:
                if token in assin_tokens:
                    is_self_op=True
                    break
        return is_self_op
    @staticmethod
    def Word_frequency_statistics(path="../data/vocab/word_frequency.pkl"):

        if os.path.isfile(path) :
            word_frequent=read_pkl(path) 
            return word_frequent
        valid_tokens=read_pkl("../data/vocab/all_candidate_tokens.pkl")
        train_raw=read_pkl("../data/train/train_raw.pkl")
        codes=train_raw['code']
        code_vars=[]
        for code in tqdm(codes):
            if isinstance(code,str):
                code=VocabModel.getCodeTokens(code)
            for token in code:
                if token in valid_tokens:
                    code_vars.append(token)
        var_cnt=Counter(code_vars)

        word_frequent = sorted(( w for (w, c) in var_cnt.items()), reverse=True)
        save_pkl(word_frequent,path)
        return word_frequent
    @classmethod
    def get_valid_tokens_name(cls,most_common=100):
        word_frequent=cls.Word_frequency_statistics()
        return word_frequent[:most_common]

if __name__ == "__main__":
    
    
    code="""

typedef struct { 
char id [ 11 ] ; 
int age ; 
} MAN ; 
void sort ( MAN * array , int nSize ) { 
int i , j ; 
for ( i = 0 ; i < nSize ; i ++ ) { 
for ( j = nSize - 1 ; j > i ; j -- ) { 
if ( ! ( ! ( array [ j ] . age >= 60 && array [ j ] . age > array [ j - 1 ] . age ) ) ) { 
MAN temp = array [ j ] ; 
array [ j ] = array [ j - 1 ] ; 
array [ j - 1 ] = temp ; 
} 
else { 
; 
} 
} 
} 
} 
int main ( ) { 
int i , n ; 
MAN * array = 0 ; 
double ava , maxgap = 0 ; 
scanf ( "%d" , & n ) ; 
array = ( MAN * ) malloc ( n * sizeof ( MAN ) ) ; 
for ( i = 0 ; i < n ; i ++ ) { 
scanf ( "%s %d" , array [ i ] . id , & ( array [ i ] . age ) ) ; 
} 
sort ( array , n ) ; 
for ( i = 0 ; i < n ; i ++ ) { 
printf ( "%s" , array [ i ] . id ) ; 
} 
free ( array ) ; 
return 0 ; 
} 

    """
    # extract information
    visitor = getCodeInfo(ori_code=code)
    visitor.show_code_info(1)
    print(visitor.stmts_line_prefix)
    print(visitor.StmtIndents)
    print()

    set_seed()
    mycfg=myConfig()
    config=mycfg.config
    logger=logging.getLogger(__name__)
    attack_config=config['new_struct']
    m=modifyStruct(code,config=attack_config)
    new_code_tokens,target_node_name=m.moveNode('temp')
    fail_times=2
    valid_times=0
