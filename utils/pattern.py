'''
Author: Zhang haha
Date: 2022-08-04 23:27:34
LastEditTime: 2023-01-23
Description: Code-related toolkits
'''
import sys
sys.path.append("..")
from data.vocab.forbidden import *
from pycparser.c_ast import * 
from pycparser.c_parser import CParser
import re
import pycparser

INDENT = " "
def extractStr(tokens):
           
    mask2token, token2mask = {}, {}
    result = []
    cnt = 0 
    for token in tokens:
        if "'" in token or '"' in token: 
            if token2mask.get(token) == None:
                mask = "<_str%d_>"%cnt
                token2mask[token] = mask
                mask2token[mask] = token
                cnt += 1
            result.append(token2mask[token])
        else:
            result.append(token)
    return result, token2mask, mask2token

def recoverStr(tokens, mask2token):
    
    result = []
    for token in tokens:
        if token.startswith("<_str"):
            result.append(mask2token[token])
        else:
            result.append(token)
    return result

def go4next(tokens, token, curIdx):
    
    n = len(tokens)
    while curIdx < n and tokens[curIdx] != token:
        curIdx += 1
    if curIdx == n:
        return -1
    else:
        return curIdx

def go4before(tokens, token, curIdx):
    
    while curIdx >= 0 and tokens[curIdx] != token:
        curIdx -= 1
    if curIdx == -1 :
        return -1
    else:
        return curIdx  

def matchIfBlocks(tokens,startToken='if',curIdx=0):
    """if else if else 匹配"""
    endToken = ""
    if startToken=='if':
        endToken=='else'
    indent = 0
    
    n = len(tokens)
    while curIdx < n:
        if tokens[curIdx] == startToken:
            indent += 1
        elif tokens[curIdx] == endToken: 
            indent -= 1
            if indent == 0:
                break
        
        curIdx += 1
    if curIdx == n:
        return -1
    else:
        return curIdx

def findBlockEnd(tokens,curIdx):
    
    n=len(tokens)
    EndIdx=-1
    if  tokens[curIdx] in ['for','while']:
       
        conditionEndIdx = go4match(tokens, "(", curIdx)
       
        if tokens[conditionEndIdx + 1] == "{":
            EndIdx = go4match(tokens, "{", conditionEndIdx)
        elif tokens[conditionEndIdx + 1] in ['for','if','do','while']:
            EndIdx=findBlockEnd(tokens,conditionEndIdx+1)
        elif tokens[curIdx+1]=='switch':
            EndIdx=go4match(tokens, "{", curIdx + 1)
            EndIdx=go4next(tokens,';',EndIdx)
        else:
            EndIdx=go4next(tokens,';',conditionEndIdx+1)
    elif tokens[curIdx]=='if':
        conditionEndIdx = go4match(tokens, "(", curIdx)
       
        if tokens[conditionEndIdx + 1] == "{":
            ifblockEndIdx = go4match(tokens, "{", conditionEndIdx + 1)
        elif tokens[conditionEndIdx + 1] in ['for','if','do','while']:
            ifblockEndIdx=findBlockEnd(tokens,conditionEndIdx+1)
        elif tokens[curIdx+1]=='switch':
            EndIdx=go4match(tokens, "{", curIdx + 1)
            EndIdx=go4next(tokens,';',EndIdx)
        else:
            ifblockEndIdx=go4next(tokens,';',conditionEndIdx+1)
        if ifblockEndIdx+1<n and tokens[ifblockEndIdx + 1]=='else':
            EndIdx=findBlockEnd(tokens,ifblockEndIdx+1)
        else: 
            EndIdx=ifblockEndIdx
    elif tokens[curIdx]=='else':
        if tokens[curIdx+1]=='if': 
            EndIdx=findBlockEnd(tokens,curIdx + 1)
        elif tokens[curIdx+1]=='{':
            EndIdx = go4match(tokens, "{", curIdx + 1)
        elif tokens[curIdx+1]=='switch':
            EndIdx=go4match(tokens, "{", curIdx + 1)
            EndIdx=go4next(tokens,';',EndIdx)
        elif tokens[curIdx+1] in ['for','if','do','while']:
            EndIdx=findBlockEnd(tokens,curIdx+1)
        else:
            EndIdx=go4next(tokens,';',curIdx+1)
    elif tokens[curIdx]=='do':
        if tokens[curIdx+1]=='{':
            doblock_endIdx=go4match(tokens, "{", curIdx+1)
            if doblock_endIdx<n and tokens[doblock_endIdx+1]=='while':
                EndIdx=go4next(tokens,';',doblock_endIdx+1)
        else:
            doblock_endIdx=go4next(tokens, ";", curIdx+1)
            if doblock_endIdx<n and tokens[doblock_endIdx+1]=='while':
                EndIdx=go4next(tokens,';',doblock_endIdx+1)

    return EndIdx

def canRemoveContentPos(attackDict):
   
    res = []
    for key in attackDict.keys():
        if key in ["count",'attack_dict_paired']:
            continue
        if attackDict[key] != []:
            for i, _ in enumerate(attackDict[key]):
                if attackDict["attack_dict_paired"].get((key, i)) is not None and attackDict["attack_dict_paired"][(key, i)] in res:
                   continue 
                res.append((key, i))

        
    return res

def removeContentFromAttackdict(attackDict, pos, listIdx=0,listIdx1=None):
    
    
    if len(attackDict[pos]) <= listIdx:
        assert False
    
    if listIdx1 is not None:
        listIdx=min(listIdx,listIdx1)
        del attackDict[pos][listIdx:listIdx+2]
    else:
        del attackDict[pos][listIdx]
    assert attackDict.get("count") is not None
    if listIdx1 is not None:
        attackDict["count"] -= 2
    else:
        attackDict["count"] -= 1

def go4match(tokens, startToken, curIdx):
    
   
    endToken = ""
    if startToken == "(":
        endToken = ")"
    elif startToken == "[":
        endToken = "]"
    elif startToken == "{":
        endToken = "}"
    else:
        assert False

    indent = 0
    n = len(tokens)
    while curIdx < n:
        if tokens[curIdx] == startToken:
            indent += 1
        elif tokens[curIdx] == endToken:
            indent -= 1
            if indent == 0:
                break
        curIdx += 1
    if curIdx == n:
        return -1
    else:
        return curIdx
def gobefore4match(tokens, endToken, curIdx):

    startToken = ""
    if endToken == ")":
        startToken = "("
    elif endToken == "]":
        startToken = "["
    elif endToken == "}":
        startToken = "{"
    else:
        assert False

    indent = 0
    n = len(tokens)
    while curIdx :
        if tokens[curIdx] == endToken:
            indent += 1
        elif tokens[curIdx] == startToken:
            indent -= 1
            if indent == 0:
                break
        curIdx -= 1
    if curIdx == 0 and tokens[curIdx]!=startToken:
        return -1
    else:
        return curIdx

def tokens2code(tokens, level=0):
    
    le_paren = 0
    idx = 0
    n = len(tokens)
    res = ""
    res += INDENT * level
    inAssign = False 
    match_else_idx=[]
    if_idxs=[]
    while idx < n:
        t = tokens[idx]
        if t=='if':
            match_else_idx=matchIfBlocks(tokens,curIdx=idx)
            if match_else_idx!=-1:
                if_idxs.append(idx)
                match_else_idx.append(match_else_idx)
        if t in ['else']:
            if tokens[idx-1]!='}':
                pre=1
                while res[-pre]==' ':
                    pre+=1
                if res[-pre]=='\n':
                    res=res[:-pre]
        res += t + " "
        if t == "(":
            le_paren += 1
        elif t == ")":
            le_paren -= 1
            if le_paren == 0:               
                inAssign = False  
                      
        elif t == ";" and le_paren == 0:    
            res += "\n"
            if idx != n - 1:
                res += INDENT * level
            inAssign = False
        elif t==":" and tokens[idx+1] in ['case','default']:
            res += "\n"
            if idx != n - 1:
                res += INDENT * level
            inAssign = False
        elif t in [";", ",", ":", "?"]:
            inAssign = False
        elif t == "{" and not inAssign:    
            res=res
            startIdx = idx + 1
            endIdx = go4match(tokens, "{", idx)
            assert endIdx!=-1
                
            res += "\n"
            res += tokens2code(tokens[startIdx: endIdx], level + 1)
            res += "\n"
            res +=INDENT * level+"}"+"\n"
            if endIdx+1 != n:
                res += INDENT * level
            idx = endIdx
        elif t == "{" and inAssign:
            idx += 1
            while idx < n:
                res += tokens[idx] + " "
                if tokens[idx] == ";":
                    res += "\n"
                    if idx != n - 1:
                        res += INDENT * level
                    inAssign = False
                    break
                idx += 1
        elif t in ["=", "enum"]:
            inAssign = True
        idx += 1
    return res

def stmts2code(stmts):
    
    res=''
    for i,stmt in enumerate(stmts):
        for token in stmt:
            res+=token+" "
        if i!=len(stmts)-1:
            res+='\n'
    return res



def getIndent(string):
   
    res = ""
    for ch in string:
        if ch in ["\t", INDENT]:
            res += ch
        else:
            break
    return res

def getIndentlevel(stmt):
    
    level=0
    for ch in stmt:
        if ch in ["\t", INDENT]:
            level+=1
        else:
            break
    if stmt.rstrip().endswith("{"):
        level+=1
    return level

def str2tokens(code):        
    
    tokens=[]
    parser = pycparser.CParser()
    parser.clex.input(code)
    t = parser.clex.token()
    while t is not None:
        tokens.append(t.value)
        t = parser.clex.token()
    return tokens


def tokens2stmts(tokens):

    tokens, token2mask, mask2token = extractStr(tokens)
    code = tokens2code(tokens)
    stmts = code.split("\n")

    stmts = ["" if stmt.rstrip() == "" else stmt.rstrip() for stmt in stmts] 

    
    newStmts = []
    for stmt in stmts:
        if stmt != "":
            newStmts.append(stmt)
    
    
    stmts = [stmt.lstrip() for stmt in stmts]
    
    '''
    struct a {              struct a {
        int i;                  int i;
        double j;   ==>         double j;
    }                       } c, d, e;
    c, d, e;
    '''
    pattern_uid = "\**[A-Za-z_][A-Za-z0-9_]*(\[[0-9]*\])*"
    pattern = "^({},\s*)*{};".format(pattern_uid, pattern_uid)
    stmts = []
    blockStack = []  
    endStruct = False # including struct & union &typedef
    for stmt in newStmts:
        if stmt.rstrip().endswith("{"):
            if endStruct:
                stmts[-1] = stmts[-1]+ " "+";"
            stmts.append(stmt)
            blockStack.append(stmt)
            endStruct=False
        elif stmt.strip() == "}":
            stmts.append(stmt)
            
            if (blockStack[-1].lstrip().startswith("struct") or\
              blockStack[-1].lstrip().startswith("union") or\
              blockStack[-1].lstrip().startswith("typedef") or\
              'struct' in blockStack[-1])and '(' not in blockStack[-1] :
               
                endStruct = True
            else:
                endStruct=False  
            blockStack.pop()
        elif endStruct:
           
            if (stmt.rstrip().endswith(";") or re.match(pattern, stmt.replace(" ",""))) and ('(' not in stmt):
                stmts[-1] = stmts[-1] + " " + stmt
            else: 
                stmts[-1] = stmts[-1]+ " "+";"
                stmts.append(stmt) 
            endStruct = False
        else:
            stmts.append(stmt)
    
    paren_n = 0 
    StmtInsPos = []
    structStack = []
    for i, stmt in enumerate(stmts):
        if stmt.rstrip().endswith("{"):
            paren_n += 1
            if stmt.lstrip().startswith("struct") or stmt.lstrip().startswith("union") or "struct" in stmt or "union" in stmt or "switch" in stmt:
                structStack.append((stmt, paren_n))
        elif stmt.lstrip().startswith("}"):
            if structStack != [] and paren_n == structStack[-1][1]:
                structStack.pop()
            paren_n -= 1
        if structStack == []:
            if i+1<len(stmts) and not stmts[i+1].lstrip().startswith("else") and not(stmts[i+1].lstrip().startswith("while") and stmts[i+1].rstrip().endswith(";")):
                
                StmtInsPos.append(i)

    indents = [getIndentlevel(stmt) for stmt in stmts]

    
    stmts=[recoverStr(stmt.strip().split(), mask2token) for stmt in stmts]
    toknesIndents=[]
    for i in range(len(stmts)):
        for j in range(len(stmts[i])):
            if stmts[i][-1]!='{':
                toknesIndents.append(indents[i])
            else:
                toknesIndents.append(indents[i]-1)
    return stmts, StmtInsPos, indents,toknesIndents
def getStmtInsPos_tokens(StmtInsPos_line,token_stmts,strict=True):
    
    res = []
    indents=[]
    cnt, indent = 0, 0 
    if not strict:
        res.append(-1)
        indents.append(0)
    for i, stmt in enumerate(token_stmts):
        cnt += len(stmt)
        if stmt[-1] == "}":
            indent = max(0,indent-1)
        elif stmt[-1] == "{" and stmt[0] not in ["struct", "union", "enum", "typedef",'static']:
            indent += 1
        if i in StmtInsPos_line:
            if stmt[-1]!="}" and\
                indent>0 and\
                stmt[0] not in ['else', 'if'] and\
                not (stmt[0]=='for' and 'if' in stmt):
                res.append(cnt-1)
                indents.append(indent)
            elif stmt[-1] == "}" and indent>1 and token_stmts[i+1][0] not in ['else',"while"] :
                res.append(cnt-1)
                indents.append(indent)
            elif stmt[-1]=="{" and stmt[0] not in ["struct", "union", "enum", "typedef","switch",'static']:
                res.append(cnt-1)
                indents.append(indent)
    if not strict and cnt-1 not in res:
        res.append(cnt-1)
        indents.append(0)
    return res,indents

def StmtInsPos(tokens, strict=True):
    
    statements, StmtInsPos, _,_ = tokens2stmts(tokens)
    code_tokens=[] 
    for tokens in statements:
        code_tokens+=tokens 
    res = []
    indents=[]
    cnt, indent = 0, 0 
    if not strict:
        res.append(-1)
        indents.append(0)
    for i, stmt in enumerate(statements):
        cnt += len(stmt)
        if stmt[-1] == "}":
            indent = max(0,indent-1)
        elif stmt[-1] == "{" and stmt[0] not in ["struct", "union", "enum", "typedef",'static']:
            indent += 1
        if i in StmtInsPos:
            if stmt[-1]!="}" and\
                indent>0 and\
                stmt[0] not in ['else', 'if'] and\
                not (stmt[0]=='for' and 'if' in stmt):
                res.append(cnt-1)
                indents.append(indent)
            elif stmt[-1] == "}" and indent>1 and statements[i+1][0] not in ['else',"while"] :
                res.append(cnt-1)
                indents.append(indent)
            elif stmt[-1]=="{" and stmt[0] not in ["struct", "union", "enum", "typedef","switch",'static']:
                res.append(cnt-1)
                indents.append(indent)
    if not strict and cnt-1 not in res:
        res.append(cnt-1)
        indents.append(0)
    return res,indents,code_tokens


def DeclInsPos(tokens):
   
    statements, _, _ = tokens2stmts(tokens)
    res = []
    cnt = 0
    res.append(-1)
    for stmt in statements:
        cnt += len(stmt)
        res.append(cnt-1)
    return res

def BrchInsPos(tokens):
    '''
    Find all possible positions to insert false branch that control flow will never reach
    '''
    return StmtInsPos(tokens)

def LoopInsPos(tokens):
    '''
    Find all possible positions to insert loop that has no effect
    '''
    return StmtInsPos(tokens)

def FuncInsPos(tokens):
    '''
    Find all possible positions to insert functions
    
    '''
    return StmtInsPos(tokens)


def InsVis(tokens, pos=[]):

    statements, _, indents = tokens2stmts(tokens)
    lens = [len(line) for line in statements]

    for pidx in pos:
        if pidx == -1:
            statements[0] = ["[____]"] + statements[0]
            continue
        cnt = 0
        for i, n in enumerate(lens):
            cnt += n
            if cnt > pidx:
                statements[i].append("[____]")
                break

    for indent, line in zip(indents, statements):
        if line[-1]=='{':
            print(str(indent)+(indent-1)*" ", end="")
        else:
            print(str(indent)+indent*" ", end="")
        print(" ".join(line))




def InsAddCandidates(insertDict, maxLen=None):
   

    res = []
    for pos in insertDict.keys():
        if pos in ["count",'added_node_numadded_node_num',"residue_space"]:
            continue
        if maxLen is None:
            res.append(pos)
        elif int(pos) < maxLen:
            res.append(pos)
    return res
def InsAattack(attackDict,pos,insertedTokenList):
   
    suc=True
    if attackDict.get(pos) is None:
        attackDict[pos]=[insertedTokenList]
    else:
        if insertedTokenList in attackDict[pos]:    # can't insert same statement
            suc = False,None
        else:
            attackDict[pos].append(insertedTokenList)
    if suc:
       
        if attackDict.get("count") is not None:
            attackDict["count"] += 1
        else:
            attackDict["count"] = 1
    return suc,(pos,len(attackDict[pos])-1)

def InsAdd(insertDict, pos, insertedTokenList):
   

    suc = True
    assert insertDict.get(pos) is not None  # this position could be inserted
    if insertedTokenList in insertDict[pos]['content']:    # can't insert same statement
        suc = False
    else:
        if isinstance(insertedTokenList,dict):
            ori_type=insertedTokenList['type']
            new_type=ori_type
            while new_type==ori_type:
                [new_type]=random.sample(type_words,1)
            insertedTokenList=[new_type,insertedTokenList["name"],';']
    insertDict[pos]['content'].append(insertedTokenList)

    if suc:
        if insertDict.get("count") is not None:
            insertDict["count"] += 1
        else:
            insertDict["count"] = 1
    return suc

# return [(pos1, 0), (pos1, 1), (pos2, 0), ...]
def InsDeleteCandidates(insertDict):
    

    res = []
    for key in insertDict.keys():
        if key == "count":
            continue
        if insertDict[key]['content'] != []:
            for i, _ in enumerate(insertDict[key]['content']):
                res.append((key, i))
    return res


def InsDelete(insertDict, pos, listIdx=0):
  
    assert insertDict.get(pos) is not None
    assert insertDict[pos]['content'] != []
    if len(insertDict[pos]['content']) <= listIdx:
        assert False
    del insertDict[pos]['content'][listIdx]
    assert insertDict.get("count") is not None
    insertDict["count"] -= 1
def GenCodeFromAttackDict(tokens, insertDict):
    
    result = []
    if insertDict.get(-1) is not None:
        for tokenList in insertDict[-1]:
            result += tokenList
    for i, t in enumerate(tokens):
        result.append(t)
        if insertDict.get(i) is not None:   # so it's a legal insertion position
            for tokenList in insertDict[i]:
                result += tokenList
    return result

def InsResult(tokens, insertDict):
  
    result = []
    if insertDict.get(-1) is not None:
        for tokenList in insertDict[-1]['content']:
            result += tokenList
    for i, t in enumerate(tokens):
        result.append(t)
        if insertDict.get(i) is not None:   # so it's a legal insertion position
            for tokenList in insertDict[i]['content']:
                result += tokenList
    return result




def getStmtEnd(tokens, optimize=3):


    statements, _, indents = tokens2stmts(tokens)
    heads = [stmt[0] for stmt in statements]
    ends = [stmt[-1] for stmt in statements]
    lens = [len(stmt) for stmt in statements]
    n = len(ends)
    endIndices = []

    # end token index for each line (single line statement)
    totalCnt = 0
    for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, indents)):
        totalCnt += cnt
        if end == ";":
            if i == n-1:
                endIndices.append(totalCnt - 1)
            elif head not in ["for", "while", "do", "if", "else", "switch"]:
                endIndices.append(totalCnt - 1)
            
            elif heads[i+1] != "else":
                endIndices.append(totalCnt - 1)
            else:
                endIndices.append(None)
        elif end == "}":
            if i == n-1:
                endIndices.append(totalCnt - 1)
            elif indents[i+1] < indent:
                endIndices.append(totalCnt - 1)
            elif heads[i+1] != "else" and heads[i+1] != "while":
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
    for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, indents)):
        
        totalCnt += cnt
        if endIndices[i] != None:
            continue
        if head not in ["for", "while", "switch"]:
            continue
        if end == "{":
            curIdx = i + 1
            indent-=1
            while curIdx < n and indents[curIdx] > indent:
                curIdx += 1
            assert curIdx < n and heads[curIdx] == "}", "[%d]%s"%(curIdx, heads[curIdx])
            endIndices[i] = endIndices[curIdx]
    for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, indents)):
        totalCnt += cnt
        if endIndices[i] != None:
            continue
        if head not in ["do"]:
            continue
        if end == "{":
            curIdx = i + 1
            indent-=1
            while curIdx < n and len(indents[curIdx]) > len(indent):
                curIdx += 1
            assert curIdx < n and heads[curIdx] == "}", "[%d]%s"%(curIdx, heads[curIdx])
            curIdx += 1
            while curIdx < n and not (indents[curIdx]==indent and heads[curIdx]=="while" and ends[curIdx]==";"):
                curIdx += 1
            assert curIdx < n and heads[curIdx] == "while", "[%d]%s"%(curIdx, heads[curIdx])
            endIndices[i] = endIndices[curIdx]
    if optimize <= 1:
        return endIndices

    # end token index for each line ("if else" statement)
    totalCnt = 0
    for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, indents)):
        totalCnt += cnt
        if endIndices[i] != None:
            continue
        if head not in ["if", "else"]:
            continue
        curTotalCnt = totalCnt
        curIdx = i
        while True:
            curIdx += 1
            while curIdx < n and len(indents[curIdx]) > len(indent):
                curTotalCnt += lens[curIdx]
                curIdx += 1
            assert curIdx < n   # because all single if/else statements have been processed in o-0
            if endIndices[curIdx] != None:
                endIndices[i] = endIndices[curIdx]
                break
    for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, indents)):
        totalCnt += cnt
        if endIndices[i] != None:
            continue
        if not (head=="}" and i+1<n and heads[i+1]=="else"):
            continue
        endIndices[i] = endIndices[i+1]
    if optimize <= 2:
        res = []
        for cnt, endIdx in zip(lens, endIndices):
            res += [endIdx] * cnt 
        return res

    # end token index for each line (left "{ }" block statement, e.g. "int main() {}" & "enum { ...; }")
    # WARNING! This WILL occur to assertion error. NO GUARANTEE!
    totalCnt = 0
    for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, indents)):
        totalCnt += cnt
        if endIndices[i] != None:
            continue
        if end == "{":
            curIdx = i + 1
            indent-=1
            while curIdx < n and indents[curIdx] > indent:
                curIdx += 1
            assert curIdx < n and heads[curIdx] == "}", "[%d]%s"%(curIdx, heads[curIdx])
            endIndices[i] = endIndices[curIdx]
    
  
    res = []
    for cnt, endIdx in zip(lens, endIndices):
        res += [endIdx] * cnt 
    return res,endIndices

def IfElseReplacePos(tokens, endPoses):
  
    pos = []
    n = len(tokens)
    for i, t in enumerate(tokens):
        if t == "if":     # only "if {} else {}", ont process "else if {}"
            ifIdx = i
            conditionEndIdx = go4match(tokens, "(", ifIdx)
            if tokens[conditionEndIdx + 1] == "{":
                ifBlockEndIdx = go4match(tokens, "{", conditionEndIdx + 1)
            else:
                ifBlockEndIdx = go4next(tokens, ";", conditionEndIdx + 1)
            if not (ifBlockEndIdx + 1 < n and tokens[ifBlockEndIdx + 1] == "else"):
                continue
            if tokens[ifBlockEndIdx + 2] == "if":   # in case of "else if {}"
                continue
            elseBlockEndIdx = endPoses[ifBlockEndIdx + 1]
            pos.append([ifIdx, conditionEndIdx, ifBlockEndIdx, elseBlockEndIdx])
    return pos

def IfElseReplace(tokens, pos):
    '''
    Description: Relpace If Else block=>excahnge the postion of if and else,
    param tokens [str]:code snippte tokens
    param pos [str]:[ifIdx, conditionEndIdx, ifBlockEndIdx, elseBlockEndIdx]
    return [list] 
    '''
    beforeIf = tokens[:pos[0]]
    codition = tokens[pos[0]+2:pos[1]]
    if tokens[pos[1]+1] == "{":
        blockIf = tokens[pos[1]+2:pos[2]]
    else:
        blockIf = tokens[pos[1]+1:pos[2]+1]
    if tokens[pos[2]+2] == "{":
        blockElse = tokens[pos[2]+3:pos[3]]
    else:
        blockElse = tokens[pos[2]+2:pos[3]+1]
    afterElse = tokens[pos[3]+1:]
    res = beforeIf + ["if", "(", "!", "("] + codition + [")", ")", "{"] + blockElse + ["}", "else", "{"] + blockIf + ["}"] + afterElse
    return res 

def IfReplacePos(tokens, endPoses):
   
    pos = []
    n = len(tokens)
    for i, t in enumerate(tokens):
        if t == "if":
            ifIdx = i
            conditionEndIdx = go4match(tokens, "(", ifIdx)
            if tokens[conditionEndIdx + 1] == "{":
                ifBlockEndIdx = go4match(tokens, "{", conditionEndIdx + 1)
            else:
                ifBlockEndIdx = go4next(tokens, ";", conditionEndIdx + 1)
            if ifBlockEndIdx + 1 < n and tokens[ifBlockEndIdx + 1] == "else":   # in case of "if {} else {}", only process "if {} xxx"
                continue
            pos.append([ifIdx, conditionEndIdx, ifBlockEndIdx])
    return pos

def IfReplace(tokens, pos):
   
    beforeIf = tokens[:pos[0]]
    condition = tokens[pos[0]+2:pos[1]]
    if tokens[pos[1]+1] == "{":
        body = tokens[pos[1]+2:pos[2]]
    else:
        body = tokens[pos[1]+1:pos[2]+1]
    afterIf = tokens[pos[2]+1:]
    # if (a) {} => if (!a); else {}
    res = beforeIf + ["if", "(", "!", "("] + condition + [")", ")", ";", "else", "{"] + body + ["}"] + afterIf
    return res 

def For2WhileReplacePos(tokens, endPoses):
   
    pos = []
    for i, t in enumerate(tokens):
        if t == "for":
            forIdx = i
            conditionEndIdx = go4match(tokens, "(", forIdx)
            if tokens[conditionEndIdx + 1] == "{":
                blockForEndIdx = go4match(tokens, "{", conditionEndIdx)
            else:
                blockForEndIdx = endPoses[conditionEndIdx + 1]
            condition1EndIdx = go4next(tokens, ";", forIdx)
            condition2EndIdx = go4next(tokens, ";", condition1EndIdx + 1)
            pos.append([forIdx, condition1EndIdx, condition2EndIdx, conditionEndIdx, blockForEndIdx])
    return pos

def For2WhileRepalce(tokens, pos):
  
    beforeFor = tokens[:pos[0]]
    condition1 = tokens[pos[0]+2:pos[1]+1]
    condition2 = tokens[pos[1]+1:pos[2]]
    if len(condition2)==0:
        condition2=['1']# while(1)
    condition3 = tokens[pos[2]+1:pos[3]] + [";"]
    if tokens[pos[3]+1] == "{":
        body = tokens[pos[3]+2:pos[4]]
    else:
        body = tokens[pos[3]+1:pos[4]+1]
    afterFor = tokens[pos[4]+1:]
    if beforeFor != [] and beforeFor[-1] in [";", "{", "}"]:
        res = beforeFor + condition1 + ["while", "("] + condition2 + [")", "{"] + body + condition3 + ["}"] + afterFor
    else:
        res = beforeFor + ["{"] + condition1 + ["while", "("] + condition2 + [")", "{"] + body + condition3 + ["}", "}"] + afterFor
    return res    

def While2ForReplacePos(tokens, endPoses):
    
    pos = []
    n = len(tokens)
    for i, t in enumerate(tokens):
        if t == "while":
            whileIdx = i
            conditionEndIdx = go4match(tokens, "(", whileIdx)
            if conditionEndIdx + 1 < n and tokens[conditionEndIdx + 1] == ";":   # in case of "do {} while ();"
                continue
            if tokens[conditionEndIdx + 1] == "{":
                blockWhileEndIdx = go4match(tokens, "{", conditionEndIdx)
            else:
                blockWhileEndIdx = endPoses[conditionEndIdx + 1]
            pos.append([whileIdx, conditionEndIdx, blockWhileEndIdx])
    return pos

def While2ForRepalce(tokens, pos):
    
    beforeWhile = tokens[:pos[0]]
    condition = tokens[pos[0]+2:pos[1]]
    if tokens[pos[1]+1] == "{":
        body = tokens[pos[1]+2:pos[2]]
    else:
        body = tokens[pos[1]+1:pos[2]+1]
    afterWhile = tokens[pos[2]+1:]
    res = beforeWhile + ["for", "(", ";"] + condition + [";", ")", "{"] + body + ["}"] + afterWhile
    return res 
def DoWhile2ForRepalace(tokens,pos):
    
    beforeDo=tokens[:pos[0]]
    if tokens[pos[0]+1]=='{':
        body=tokens[pos[0]+2:pos[1]]
    else:
        body=tokens[pos[0]+1:pos[1]+1]
    condition=tokens[pos[2]+2:pos[3]]
    afterWhile=tokens[pos[3]+2:]# do{}while();
    res=beforeDo+ ["for", "(", ";"] + condition + [";", ")", "{"] + body + ["}"] + afterWhile
    return res

class CASTGraph_simple():
    
    def __init__(self,stmts=None):
       
       
        self.CopyInsertPos={}
        self.stmts=stmts
        self.level_decl={}
       
        self.Assign={}
       
        self.Decl={}
      
        self.Func={}
        # while
        self.while_blcok={}
        # for
        self.for_block={}
       
        self.forbidden_tokens = key_words + ops + macros + special_ids
    
  
    def dfs_ast(self,node):
        if node.coord:
            line=node.coord.line-1
            s_pos=node.coord.column-1
        cand_end_flag=[',',';']
        block_match={"{":"}",'"':'"',"(":")","'":"'"}
        if isinstance(node,Assignment):
            end_pos=node.rvalue.coord.column-1
            equal_pos=end_pos-1
            
            if "if" in self.str_stmts[line]:
                return 
            while self.str_stmts[line][equal_pos]!='=':
                equal_pos-=1
            left_value= self.str_stmts[line][s_pos:equal_pos].strip().split(" ")
            block=[]
            while (self.str_stmts[line][end_pos] not in cand_end_flag) or len(block)!=0 :
                if len(block)==0 and self.str_stmts[line][end_pos]==")":# in case of for ( int i = 6 ; i <= n ; i += 2 )
                    return 
                if len(block) and block_match[block[-1]]==self.str_stmts[line][end_pos]:
                    block.pop(-1)
                    end_pos+=1
                    continue
                if self.str_stmts[line][end_pos] in  list(block_match.keys()): 
                    if "'" not in block and  '"' not in block:
                        block.append(self.str_stmts[line][end_pos])
                if end_pos==len(self.str_stmts[line]):
                    break
                end_pos+=1
            if end_pos<len(self.str_stmts[line]):
                assign=(self.str_stmts[line][s_pos:end_pos]).strip().split(" ")
                can_copyed=True
                for token in left_value:
                    if token  in self.forbidden_tokens or token not in assign:
                        can_copyed=False
                if can_copyed:
                    self.Assign[(line,end_pos-1)]={"leftvalue":left_value,"assign":assign}
                   
        if node.coord and type(node) == ArrayDecl and self.str_stmts[line][-1]!='{' and "=" not in self.str_stmts[line] :
            end_pos=s_pos+1
            block=[]
            while(self.str_stmts[line][end_pos] not in cand_end_flag or len(block)!=0):
                # B[10]={1,2,3}这种情况 还有char b [ 2 ] [ 10 ] = { { "Sun." } , { "Mon." } }   
                if len(block) and block_match[block[-1]]==self.str_stmts[line][end_pos]:
                    block.pop(-1)
                    end_pos+=1
                    continue
                if self.str_stmts[line][end_pos] in  list(block_match.keys()): 
                    block.append(self.str_stmts[line][end_pos])

                # ( * p ) [ 100 ] = ( int ( * ) [ 100 ] ) calloc ( 100 , 100 * sizeof ( int ) )这种情况
                if self.str_stmts[line][end_pos]==")": 
                    while self.str_stmts[line][s_pos ]!="(":
                        s_pos-=1
                end_pos+=1
            decl=(" , "+self.str_stmts[line][s_pos:end_pos]).split(" ")
            self.CopyInsertPos[(line,end_pos-1)]=decl
            return 
        if node.coord and type(node) == TypeDecl and self.str_stmts[line][-1]!='{' and "(" not in self.str_stmts[line]:
           
            end_pos=s_pos+1
            
            cur_decl={}
            if isinstance(node.type,IdentifierType):
                cur_decl['type']=node.type.names[0]
            else:cur_decl['type']=None
            cur_decl['name']=node.declname
            if self.level_decl.get(self.indent_levels[line]) is None:
                self.level_decl[self.indent_levels[line]]=[cur_decl]
            else: 
                self.level_decl[self.indent_levels[line]].append(cur_decl)
            block=[]
            while(self.str_stmts[line][end_pos] not in cand_end_flag or len(block)!=0):
               
                if len(block) and block_match[block[-1]]==self.str_stmts[line][end_pos]:
                    block.pop(-1)
                    end_pos+=1
                    continue
                if self.str_stmts[line][end_pos] in  list(block_match.keys()): 
                    block.append(self.str_stmts[line][end_pos])
                    end_pos+=1
                    continue

                # ( * p ) [ 100 ] = ( int ( * ) [ 100 ] ) calloc ( 100 , 100 * sizeof ( int ) )
                if self.str_stmts[line][end_pos]==")": 
                    while self.str_stmts[line][s_pos]!="(":
                        s_pos-=1
                end_pos+=1
            decl=(" , "+self.str_stmts[line][s_pos:end_pos]).split(" ")
            self.CopyInsertPos[(line,end_pos-1)]=decl
            return 
        if node.coord and type(node) == FuncDecl:
            if self.str_stmts[line].strip().endswith(";") and "{" not in self.str_stmts[line] :
                self.CopyInsertPos[(line,len(self.str_stmts[line]))]= (" "+self.str_stmts[line].strip()).split(" ")
                return 
        for _, child_ in node.children():# [type,Node]
            self.dfs_ast(child_)
    def visit(self,root,str_stmts=None,stmts=None,indent_levels=None):
        
        self.ast=root
        self.str_stmts=str_stmts
        self.indent_levels=indent_levels
        self.dfs_ast(self.ast)
    def get_decl_value(node):
        pass

def CopyInsPos(tokens):
   
    stmts,StmtInsPos_line,indents=tokens2stmts(tokens)
    
    
    nor_code=stmts2code(stmts)
    str_stmts=nor_code.split("\n") 
    
    ast=CParser().parse(nor_code)
    cs=CASTGraph_simple()
    cs.visit(ast,str_stmts,indent_levels=indents)
    copy_insert_dic=changeInsertPos(cs.CopyInsertPos,stmts,str_stmts,indents)
    decl_insert_dict=getDeclInsertDict(cs.level_decl,stmts,StmtInsPos_line)
    return copy_insert_dic,decl_insert_dict
# only support one piece of data each time: x is idx-list
def getCopyInsertDict(x_raw):
   
   
    stmts,StmtInsPos_line, indents,_=tokens2stmts(x_raw)
    
    nor_code=stmts2code(stmts)
    str_stmts=nor_code.split("\n") 
   
    ast=CParser().parse(nor_code)
    cs=CASTGraph_simple()
    cs.visit(ast,str_stmts,indent_levels=indents)
    new_insert_dic=changeInsertPos(cs.CopyInsertPos,stmts,str_stmts,indents=indents)
    return new_insert_dic


def changeInsertPos(InsertPos,stmts,loc_stmts,indents):
   
    new_insert_dic={}
    for key in InsertPos.keys():
        line=key[0]
        insert_token_id=strloc2tokenloc(key,stmts,loc_stmts)
       
        
        copy_code=InsertPos[key]
        # loc_stmts[key[0]][key[1]]
        if key not in copy_code:
            pos={'content':InsertPos[key],'level':indents[line]}
            new_insert_dic[insert_token_id]=pos
    return new_insert_dic

def strloc2tokenloc(strloc,stmts,loc_stmts):
    
    loc_id=0
    for i in range(strloc[0]):
       loc_id+=len(stmts[i])
   
    is_start=True
    j=0
    while j <(strloc[1]):
        if loc_stmts[strloc[0]][j]!=' ' and is_start:
            is_start=False
        
        if loc_stmts[strloc[0]][j]==' ' and  not is_start:
            while(loc_stmts[strloc[0]][j]==' ' and j <(strloc[1])-1):
                j+=1
            loc_id+=1
            
            if  loc_stmts[strloc[0]][j]=='"':
                if j+1 <strloc[1] and j+2<strloc[1]:
                    if loc_stmts[strloc[0]][j+1]==' ' and loc_stmts[strloc[0]][j+2]=='"':
                        j=j+2
                    continue
            if  loc_stmts[strloc[0]][j]=="'":
                if j+1 <strloc[1] and j+2<strloc[1]:
                    if loc_stmts[strloc[0]][j+1]==' ' and loc_stmts[strloc[0]][j+2]=="'":
                        j=j+2
                    continue
        j+=1
    return loc_id
def DeclDomainInsPos(stmts,StmtInsPos_line, strict=True):
    
    res = []
    indents=[]
    cnt, indent = 0, 0 
    if not strict:
        res.append(-1)
        indents.append(0)
    for i, stmt in enumerate(stmts):
        cnt += len(stmt)
        if stmt[-1] == "}":
            indent = max(0,indent-1)
        elif stmt[-1] == "{" and stmt[0] not in ["struct", "union", "enum", "typedef"]:
            indent += 1
        if i in StmtInsPos_line:
            if stmt[-1]!="}" and\
                indent>0 and\
                stmt[0] not in ['else', 'if'] and\
                not (stmt[0]=='for' and 'if' in stmt):
                res.append(cnt-1)
                indents.append(indent)
            elif stmt[-1] == "}" and indent>1 and stmts[i+1][0] not in ['else',"while"] :
                res.append(cnt-1)
                indents.append(indent)
            elif stmt[-1]=="{" and stmt[0] not in ["struct", "union", "enum", "typedef","switch"]:
                res.append(cnt-1)
                indents.append(indent)
    if not strict and cnt-1 not in res:
        res.append(cnt-1)
        indents.append(0)
    return res,indents
def getDeclInsertDict(tokens):

    stmts,StmtInsPos_line, indents,_=tokens2stmts(tokens)
   
    nor_code=stmts2code(stmts)
    str_stmts=nor_code.split("\n") 
  
    ast=CParser().parse(nor_code)
    cs=CASTGraph_simple()
    cs.visit(ast,str_stmts,indent_levels=indents)
   
    attack_domain=getDeclInsertLevel(cs.level_decl)
    
    DeclInsertDict={}
    insert_poses,indents=DeclDomainInsPos(stmts,StmtInsPos_line,False)
    for i in range(len(insert_poses)):
        level=indents[i]
        if attack_domain.get(level):
            
            pos={'content':[decl for decl in attack_domain[level]],'level':level}
            DeclInsertDict[insert_poses[i]]=pos
            
    return DeclInsertDict
def counts_old_attack_dict(attack_dict):
    
    change_nums=0
    for key,item in attack_dict.items():
        if key!="count":
            for add_content in attack_dict[key]['content']:
                change_nums+=len(add_content)
    return change_nums
def counts_rename_changes(old_tokens,new_tokens):
    change_nums=0
    for ot,nt in zip(old_tokens,new_tokens):
        if ot!=nt:
            change_nums+=1
    return change_nums       
def get_new_decl(decl_insert):
    
    ori_type=decl_insert['type']
    new_type=ori_type
    while new_type==ori_type:
        [new_type]=random.sample(type_words,1)
    return [new_type,decl_insert["name"],';']
def getDeclInsertPos(level_decl):
   
    res = []
    for key in level_decl.keys():
        if level_decl[key] != []:
            for attack_level in range(key):
                for i, decl_insert in enumerate(level_decl[key]):
                    res.append((attack_level, decl_insert))
    return res
def getDeclInsertLevel(level_decl):

    res={}
    for key in level_decl.keys():
        if level_decl[key] != []:
            for attack_level in range(key):
                for decl_insert in level_decl[key]:
                    if res.get(attack_level) is None:
                        res[attack_level]=[decl_insert]
                    else: res[attack_level].append(decl_insert)
    return res
def attack_level_decl(StmtInsPos_line,indents,target_level):
  
    valid_pos=[]
    for i in StmtInsPos_line:
        if indents[i]==target_level and i !=len(indents)-1 :
            valid_pos.append(i)
   
    return valid_pos   


def show_pos_end_ex():
    code=""
    
    tokens=str2tokens(code)
    InsVis(tokens)
    stmts, _, indents = tokens2stmts(tokens)
    token_end_pos,stmt_end_pos=getStmtEnd(tokens)
    
    for stmt, endIdx, indent in zip(stmts, stmt_end_pos, indents):
        if endIdx != None:
            print("{:30}".format(" ".join(tokens[endIdx-3:endIdx+1])+"!!=>"+tokens[endIdx]), end="")
        else:
            print("{:30}".format(" "), end="")
        if stmts[-1]=='{':
            print(str(indent)+(indent-1)*" ", end="")
        else:
            print(str(indent)+indent*" ", end="")
        print(" ".join(stmt)) 

if __name__ == "__main__":
    
    # test loc attack pos

    code="""
    int main ( ) 
    { 
     
    int call ( int month , int day ); 
    int k , year , month , day , num ;  
    if ( k == 1 ) { 
        num = call ( month , day ) ; 
        } 
   
    return 0 ; 
    } 
    
    int cal ( int month , int day ) 
    { 
        int k =1; 
    }  

    """
    parser = pycparser.CParser()
    parser.clex.input(code)
    tokens_code=[]
    
    t = parser.clex.token()
    while t is not None:
        tokens_code.append(t.value)
        t=parser.clex.token()
    stmts, StmtInsPos_line, indents,_=tokens2stmts(tokens_code)

    code=stmts2code(stmts)
    loc_stmts=code.split("\n") 
    ast=CParser().parse(code)
    cs=CASTGraph_simple()
    cs.visit(ast,loc_stmts,stmts,indents)
    copy_attack_poses=getCopyInsertDict(tokens_code)
    decl_insert_dict=getDeclInsertDict(tokens_code)
    
    print(copy_attack_poses)
    