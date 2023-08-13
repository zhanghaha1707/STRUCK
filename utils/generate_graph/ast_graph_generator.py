'''
Author: Zhang
Date: 2022-08-11
LastEditTime: 2023-01-23
Description: consrtruct code graph 

'''



from collections import defaultdict
from pycparser.c_parser import CParser
from pycparser.c_ast import  NodeVisitor
from pycparser.c_ast import *
# edge type
EDGE_TYPE = {
    'child': 0,
    'NextToken': 1,
    'last_lexical': 2,
    'last_use': 3,
    'last_write': 4,
    'computed_from': 5,
    'return_to': 6,
}
# node type
NODE_TYPE = {
    'non_terminal': 0,
    'terminal': 1,
    'identifier': 2
}

class CAstGraphGenerator(NodeVisitor):
    def __init__(self):
        self.ast=None
        
        self.node_id=0                  
        self.node_label={}             
        self.node_type={}               
        self.terminal_path = []         
        self.graph=defaultdict(set)     
        

       
        self.use_ast=True               
        self.syntactic_only = False          
        self.identifier_only = False    
        self.is_return = False                  
        self.isEnd=True                
       
        self.node_level=0
        
        self.need_semicolon=False
        
        self.Ptrnum=0
       
        self.parent = None              
        self.previous_token = None      
        self.last_lexical={}            
        self.last_use=defaultdict(lambda:set())                
        self.last_write=defaultdict(lambda:set())               
        self.previous_access = defaultdict(lambda: [set(), set()]) 
        self.current_function = None    
        
        self.assign_context = None      
        self.lvalue=False                
        self.is_revisit = False         
        
    def __add_edge(self,nid,label=None,edge_type='child'): 
       
        if edge_type == 'child' and self.parent is not None and self.use_ast and not self.is_revisit :
            
            self.graph[(self.parent, nid)].add('child') 

        if edge_type == 'NextToken' and self.previous_token is not None and not self.is_revisit:
            
            self.graph[(self.previous_token, nid)].add('NextToken')
        
        
        if edge_type == 'last_lexical' and label in self.last_lexical and not self.is_revisit:
            
            self.graph[(nid, self.last_lexical[label])].add('last_lexical')
        
        
        if edge_type == 'last_use' and not self.syntactic_only:
            for use in self.last_use[label]:
                self.graph[(nid, use)].add('last_use')
        if edge_type == 'last_write' and not self.syntactic_only and label in self.last_write:
            for write in self.last_write[label]:
                self.graph[(nid, write)].add('last_write')
        if edge_type == 'computed_from' and self.lvalue\
                                        and self.assign_context is not None:
           
            for rvalue in self.assign_context:
                self.graph[(nid,rvalue)].add('computed_from')
        
        if edge_type == 'return_to' and not self.syntactic_only \
                                        and self.is_return and self.use_ast \
                                        and self.current_function is not None:
            self.graph[(nid, self.current_function)].add('return_to')
        
         
    def __create_node(self,label,node_type):
       
        self.node_label[self.node_id] = label 
        self.node_type[self.node_id] = node_type
        if (node_type == NODE_TYPE['terminal'] or node_type == NODE_TYPE['identifier']) \
           and self.node_id not in self.terminal_path:
            self.terminal_path.append(self.node_id)
        self.node_id += 1      
        return self.node_id - 1 
    def judge_Need_semicolon(self,id):
    
        if self.node_label[id-1]==';':
            return False
        else:
            return True
          
           
    def revisit(self,node,start_node_id):
       
        old_id, old_parent, old_last_lexical, old_previous =  \
                        (self.node_id, self.parent, self.last_lexical, self.previous_token)
        self.node_id = start_node_id     
        self.is_revisit = True  
        self.visit(node)     
        end_id = self.node_id  
        self.is_revisit = False
       
        self.node_id, self.parent, self.last_lexical, self. previous_token = \
                        (old_id, old_parent, old_last_lexical,  old_previous)
        return end_id
    
    def non_terminal(self, node):
       
        if self.use_ast:
            nid = self.__create_node(node.__class__.__name__, NODE_TYPE['non_terminal'])
            self.__add_edge(nid,edge_type='child')
            self.parent = nid
        else:
            pass
   
    def terminal(self,label):
      
        if not self.identifier_only:
            nid = self.__create_node(label, NODE_TYPE['terminal'])
            
            self.__add_edge(nid, edge_type='child')
            self.__add_edge(nid, edge_type='NextToken')
            self.__add_edge(nid, edge_type='return_to')
            if not self.is_revisit:
                self.previous_token = nid
           
            if self.assign_context is not None and not self.lvalue:
                
                self.assign_context.add(nid)
        else:
            pass
    def identifier(self, label):
     
        nid = self.__create_node(label, NODE_TYPE['identifier'])
       
        self.__add_edge(nid, edge_type='child')
        self.__add_edge(nid, edge_type='NextToken')
        self.__add_edge(nid, label=label, edge_type='last_lexical')

        self.__add_edge(nid, label=label, edge_type='last_use')
        self.__add_edge(nid, label=label, edge_type='last_write')
        self.__add_edge(nid, label=label, edge_type='computed_from')

        if not self.is_revisit:
            self.previous_token = nid
            self.last_lexical[label] = nid
        if not self.syntactic_only:
            self.last_use[label] ={nid} 
            if self.lvalue:
                self.last_write[label]={nid}
        
        if self.assign_context is not None and not self.lvalue:
           
            self.assign_context.add(nid)
     

    def __copy_semantic_dict(self,last_use_dict,last_write_dict):
        
        return defaultdict(last_use_dict.default_factory,
                           {key: use.copy() for key, use in last_use_dict.items()}),\
               defaultdict(last_write_dict.default_factory,
                           {key: write.copy() for key, write in last_write_dict.items()})
    def __add_context(self,new_last_use,new_last_write):
        
         return defaultdict(self.last_use.default_factory,{label:set.union(new_last_use[label],self.last_use[label]) for label in self.last_use.keys()}),\
             defaultdict(self.last_write.default_factory,{label:set.union(new_last_write[label],self.last_write[label]) for label in self.last_write.keys()})

    def __enter_branching(self):
        return self.__copy_semantic_dict(self.last_use,self.last_write),self.__copy_semantic_dict(self.last_use,self.last_write)
    def __new_branch(self, save_last_use,save_last_write, new_last_use,new_last_write,if_and_else=False):
     
       
        if if_and_else:
            new_last_use,new_last_write=self.__copy_semantic_dict(self.last_use,self.last_write)
        else:
            new_last_use,new_last_write=self.__add_context(new_last_use,new_last_write)
        self.last_use,self.last_write=self.__copy_semantic_dict(save_last_use,save_last_write)
        return new_last_use,new_last_write
    def __leave_branching(self,new_last_use,new_last_write):
     
        self.last_use,self.last_write = self.__add_context(new_last_use,new_last_write)
    

  
    
    def visit_FileAST(self,node):
        
        self.non_terminal(node)
        self.generic_visit(node)
    def visit_FuncDef(self,node):
       
        save_parent=self.parent
        top_function=self.current_function 
        self.current_function=self.node_id
        self.non_terminal(node)
        self.visit(node.decl)   
        self.node_level+=1
        self.visit(node.body)

        self.current_function = top_function
        self.parent=save_parent         
    def visit_Decl(self,node):
       
     

        save_parent=self.parent
        self.non_terminal(node)
        if node.init:
            self.syntactic_only = True
       
        for qual in node.quals:
            self.terminal(qual)
       
        for funcspec in node.funcspec:
            self.terminal(funcspec)
      
        for storage in node.storage:
            self.terminal(storage)
        
        assign_start_node_id=self.node_id
        self.visit(node.type)
        
        if node.init:
            self.terminal('=')
            self.syntactic_only = False
          
            self.assign_context = set()
            self.visit(node.init)
            self.lvalue = True
            self.revisit(node.type,assign_start_node_id) 
            self.assign_context=None
            self.lvalue = False
        if not isinstance(node.type,FuncDecl) and  self.need_semicolon :
            self.terminal(';')
        self.parent=save_parent           
    def visit_InitList(self,node):
       
        save_parent=self.parent
        self.terminal('{')
        for expr in node.exprs[:-1]:
            self.visit(expr)
            self.terminal(',')
        self.visit(node.exprs[-1])
        self.terminal('}')
        self.parent=save_parent 
    def visit_FuncDecl(self,node):
      
        save_parent=self.parent
      
        self.non_terminal(node)
        
        self.visit(node.type)
       
        self.terminal('(')
        if node.args:
            self.visit(node.args)
        self.terminal(')')
        self.parent=save_parent 
    def visit_ParamList(self,node):
        
        self.need_semicolon=False
        for child in node.params[:-1]:
            self.visit(child)
            self.terminal(",")
        self.visit(node.params[-1])
        self.need_semicolon=True
    def visit_Typedef(self,node):
       
        save_parent=self.parent
        self.non_terminal(node)
        for stor in node.storage:
            self.terminal(stor)
        for qual in node.quals:
            self.terminal(qual)
        
        self.visit(node.type)
       
        self.parent=save_parent
    def visit_TypeDecl(self,node):   
        
        save_parent=self.parent
        self.non_terminal(node)
        if isinstance(node.type,IdentifierType):
            type_name=node.type.names[0]
            self.terminal(type_name)
        else:self.visit(node.type)
        label=node.declname
        
        if label is not None:
            for i in range(self.Ptrnum):
                self.terminal('*')
            self.identifier(label)
        self.parent=save_parent

        
    
    def visit_ArrayDecl(self,node):
        
        save_parent=self.parent
        self.non_terminal(node)
        for qual in node.dim_quals:
            self.terminal(qual)
        self.visit(node.type)
        self.terminal('[')
        if node.dim is not None:
            self.visit(node.dim)
        self.terminal(']')
        self.parent=save_parent
    def visit_ArrayRef(self,node):
     
        save_parent=self.parent
        self.non_terminal(node)
        self.visit(node.name)
        self.terminal('[')
        self.visit(node.subscript)
        self.terminal(']')
        self.parent=save_parent
    def visit_Union(self,node):
        save_parent=self.parent
        self.non_terminal(node)
        self.terminal('union')
        if node.name is not None:
            self.identifier(node.name)
            for i in range(self.Ptrnum):
                self.terminal('*')
        if node.decls is not None:
            self.terminal("{")
            for decl in node.decls:
                self.visit(decl)
            self.terminal("}")
        self.parent=save_parent

    def visit_Struct(self,node):
     
        save_parent=self.parent
        self.non_terminal(node)
        self.terminal('struct')
        if node.name is not None:
            self.identifier(node.name)
            for i in range(self.Ptrnum):
                self.terminal('*')
        if node.decls is not None:
            self.terminal("{")
            for decl in node.decls:
                self.visit(decl)
            self.terminal("}")
        self.parent=save_parent
    def visit_StructRef(self,node):
        
       
        save_parent=self.parent
        self.non_terminal(node)
        self.visit(node.name)
        self.terminal(node.type)
        self.terminal(node.field.name)
        
        self.parent=save_parent
    def visit_Compound(self,node):
       
        save_parent=self.parent
        self.non_terminal(node)
        
        self.terminal('{')
        
        for child in node.children():
            
            self.visit(child[1])
            if self.need_semicolon and self.judge_Need_semicolon(self.node_id):
                self.terminal(';')
        self.terminal('}')
        self.parent=save_parent
    def visit_CompoundLiteral(self,node):
    
        pass

    def visit_Constant(self,node):
       
        save_parent=self.parent
      
        self.terminal(str(node.value))
        self.parent=save_parent
    def visit_PtrDecl(self,node):
       
        save_parent=self.parent
        self.Ptrnum=1
        for qual in node.quals:
            self.terminal(qual)
        if isinstance(node.type,PtrDecl):
            node=node.type
            self.Ptrnum+=1
        
        self.visit(node.type) 
        self.Ptrnum=0   
      
        self.parent=save_parent

    def visit_ID(self,node):
     
        save_parent=self.parent
       
        self.identifier(node.name)
        self.parent=save_parent
    def visit_Assignment(self,node):
       
        save_parent=self.parent
        self.non_terminal(node)
        assign_start_node_id=self.node_id
        self.syntactic_only = True
        self.visit(node.lvalue)
        
        self.terminal(node.op)
        self.syntactic_only = False 
        self.assign_context = set()
        self.visit(node.rvalue)
        self.lvalue = True
        
        self.revisit(node.lvalue,assign_start_node_id) 
        self.assign_context=None
        self.lvalue=False 

        self.parent=save_parent

    def visit_BinaryOp(self,node):
      
        save_parent=self.parent
        self.non_terminal(node)

        self.visit(node.left)
        self.terminal(node.op)
      
        
        self.visit(node.right)
        self.parent=save_parent
    def visit_Cast(self,node):
    
        save_parent=self.parent
        self.terminal('(')
        self.visit(node.to_type)
       
        self.terminal(')')
        self.visit(node.expr)
        self.parent=save_parent
    def visit_Typename(self,node):
        save_parent=self.parent
        self.non_terminal(node)
        for quals in node.quals:
            self.terminal(quals)
        self.visit(node.type)
        self.parent=save_parent
    def visit_FuncCall(self,node):
      
        save_parent=self.parent
  
        self.visit(node.name)
        self.terminal('(')
        if node.args is not None:
            self.visit(node.args)
        self.terminal(')')
      
        self.parent=save_parent

    def visit_ExprList(self,node):
    
        for child in node.children()[:-1]:
            self.visit(child[1])
            self.terminal(',')
        self.visit(node.children()[-1][1])
        

    def visit_UnaryOp(self,node):
   
        if node.op.startswith('p'):
            self.visit(node.expr) 
            self.terminal(node.op[1:])
        else:
            self.terminal(node.op)
            if node.op =='sizeof':
                self.terminal("(")
            self.visit(node.expr) 
            if node.op =='sizeof':
                self.terminal(")")
 
    def visit_TernaryOp(self,node):
       
        save_parent=self.parent
        self.isEnd=False
        self.non_terminal(node)
        self.terminal('(')
        self.visit(node.cond)
        self.terminal(")")
        self.terminal('?')
       
        self.visit(node.iftrue)
        
        self.terminal(":")
        self.visit(node.iffalse)
        
        self.isEnd=True
        self.parent=save_parent
        

    def visit_If(self,node):
       
        save_parent=self.parent
        self.non_terminal(node)
        self.terminal('if')
        self.terminal('(')
        self.visit(node.cond)
        self.terminal(')')
        
        (save_last_use,save_last_write),( branched_last_use,branched_last_write)=self.__enter_branching() 
        self.visit(node.iftrue)
      
        if node.iffalse: 
           
            temp_node=node.iffalse
            if_and_else=True 
            while "iftrue" in dir(temp_node):
                if temp_node.iffalse is None:
                    if_and_else=False
                    break
                temp_node=temp_node.iffalse

            
            branched_last_use,branched_last_write=self.__new_branch(save_last_use,save_last_write,branched_last_use,branched_last_write,if_and_else)
            self.terminal('else')
            self.visit(node.iffalse)
           
        self.__leave_branching(branched_last_use,branched_last_write)
        self.parent=save_parent  
    def visit_Switch(self,node):
       
        save_parent=self.parent
        self.non_terminal(node)
        self.terminal('switch')
        self.terminal("(")
        self.visit(node.cond)
        self.terminal(')')
        self.visit(node.stmt)
        self.parent=save_parent
    def visit_Case(self,node):
     
        save_parent=self.parent
        self.non_terminal(node)
        self.terminal('case')
        self.visit(node.expr)
        self.terminal(":")
        for child in node.stmts:
            self.visit(child)
        self.parent=save_parent
    def visit_Default(self,node):
        
        save_parent=self.parent
        self.non_terminal(node)
        self.terminal('default')
        self.terminal(":")
        for child in node.stmts:
            self.visit(child)
        self.parent=save_parent
    def visit_EllipsisParam(seld,node):
     
        pass
    def visit_Enum(self,node):
        save_parent=self.parent
        self.non_terminal(node)
        self.terminal('enum')
        self.identifier(node.name)
        self.terminal("{")
        self.visit(node.values)
        self.terminal('}')
        self.parent=save_parent
    def visit_EnumeratorList(self,node):
       
        save_parent=self.parent
        for child in node.enumerators:
            self.visit(child)
        self.parent=save_parent
    def visit_Enumerator(self,node):
     
        self.identifier(node.name)
        if node.value is not None:
            self.terminal('=')
            self.visit(node.value)
    def visit_For(self,node):
      
        save_parent=self.parent
        self.non_terminal(node)
        self.terminal('for')
        self.terminal("(")
        (save_last_use,save_last_write),( branched_last_use,branched_last_write)=self.__enter_branching()
        if node.init is not None:
            self.visit(node.init)
        else:self.terminal(";")
        if node.cond is not None:
            self.visit(node.cond)
        self.terminal(";")
        if node.next is not None:
            self.visit(node.next)
        self.terminal(")")
        self.visit(node.stmt)
        self.__leave_branching(branched_last_use,branched_last_write)
        self.parent=save_parent
    def visit_DeclList(self,node):
        
        save_parent=self.parent
        self.parent=save_parent
    def visit_While(self, node):
       
        save_parent=self.parent
        self.non_terminal(node)
        self.terminal('while')
        self.terminal("(")
        self.visit(node.cond)
        self.terminal(")")
        (save_last_use,save_last_write),( branched_last_use,branched_last_write)=self.__enter_branching()
        self.visit(node.stmt)
        self.__leave_branching(branched_last_use,branched_last_write)
        self.parent=save_parent
    def visit_DoWhile(self,node):
        save_parent=self.parent
        self.non_terminal(node)
        self.terminal('do')
        (save_last_use,save_last_write),( branched_last_use,branched_last_write)=self.__enter_branching()
        self.visit(node.stmt)
        branched_last_use,branched_last_write=self.__new_branch(save_last_use,save_last_write,branched_last_use,branched_last_write)
        self.terminal("while")
        self.terminal("(")
        self.visit(node.cond)
        self.terminal(")")
        self.__leave_branching(branched_last_use,branched_last_write)
        self.parent=save_parent
    def visit_Break(self,node):
      
        self.terminal("break")
        if self.need_semicolon:
            self.terminal(";")
    def visit_Continue(self,node):
        self.terminal("continue")
        if self.need_semicolon:
           self.terminal(";")
    def visit_Return(self,node):
       
        save_parent=self.parent
        self.non_terminal(node)
        if node.expr is not None:
            self.is_return = True
            self.terminal('return')
            self.visit(node.expr)
            self.is_return = False
        else:
            self.is_return = True
            self.terminal('return')
            self.is_return = False
        if self.need_semicolon:
            self.terminal(';')
        self.parent=save_parent
    
    def generic_visit(self,node):
        
        for child in node.children():
            self.visit(child[1])    

def visualize_graph(G,labels=None,color=None):
    import networkx as nx
    import matplotlib.pylab as plt
    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
    nx.draw(G)
    plt.savefig('pygG.png')

if __name__ == "__main__":
    
    code="""
    int a;
        
    """
    # with open("./erro.txt",'r') as f:
    #     code=f.read()
    root=CParser().parse(code)
    
    visitor = CAstGraphGenerator()
    visitor.visit(root)
    node_label=visitor.node_label
    edge_list = [(origin, destination,t)
                        for (origin, destination), edges
                        in visitor.graph.items() for t in edges]
    graph_node_labels = [str(nid)+'-'+str(label).strip() for (nid, label) in sorted(visitor.node_label.items())]
    graph = {"edges": edge_list, "backbone_sequence": visitor.terminal_path, "node_labels": graph_node_labels}
    print(graph)
    child_edge=[(origin, destination,t) 
                        for (origin, destination), edges
                        in visitor.graph.items() for t in edges if t =='child']
    netx_edge=[(origin, destination,t) 
                        for (origin, destination), edges
                        in visitor.graph.items() for t in edges if t =='NextToken']
    computed_from=[(origin, destination,t) 
                        for (origin, destination), edges
                        in visitor.graph.items() for t in edges if t =='computed_from']
    last_acsess_edges= [(origin, destination,t) 
                        for (origin, destination), edges
                        in visitor.graph.items() for t in edges if t.startswith('last')]                   
 
    
    for i in  visitor.terminal_path:
        print(node_label[i],end=" ")
    print()
  
    print(last_acsess_edges)
    source=[]
    dest=[]
    edge_types={}
    i=0
    for (s,t,edge_type) in edge_list:
        source.append(s)
        dest.append(t)
        edge_types[i]=edge_type
        i+=1
    from torch_geometric.data import Data
    from torch_geometric.utils import to_networkx
    import networkx as nx
    import matplotlib.pylab as plt
    import torch
    nodes=list(node_label.keys())
    g=Data(nodes,edge_index=torch.stack((torch.tensor(source),torch.tensor(dest)),dim=0))
    G = to_networkx(g, to_undirected=False)
   
    visualize_graph(G,edge_types)