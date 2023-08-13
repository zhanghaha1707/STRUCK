'''
Author: Zhang
Date: 2022-07-19
LastEditTime: 2022-07-19
FilePath: /code_struct_attacks/code-graphcodebert/parser/tree-sitter-c/build.py
Description: 

'''
from tree_sitter import Language, Parser

Language.build_library(
  # Store the library in the `build` directory
  'my-languages.so',

  # Include one or more languages
  [
    'tree-sitter-c',
  ]
)
