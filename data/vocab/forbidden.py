'''
Author: Zhang haha
Date: 2022-08-03 00:47:41
LastEditTime: 2023-01-20
Description: Need to avoid replacing token

'''
type_words=["char","int","float", "double"]

key_words= ["auto", "break", "case", "char", "const", "continue",
                "default", "do", "double", "else", "enum", "extern",
                "float", "for", "goto", "if", "inline", "int", "long",
                "register", "restrict", "return", "short", "signed",
                "sizeof", "static", "struct", "switch", "typedef",
                "union", "unsigned", "void", "volatile", "while",
                "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
                "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
                "_Thread_local", "__func__"]

ops= ["...", ">>=", "<<=", "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=",
        ">>", "<<", "++", "--", "->", "&&", "||", "<=", ">=", "==", "!=", ";",
        "{", "<%", "}", "%>", ",", ":", "=", "(", ")", "[", "<:", "]", ":>",
        ".", "&", "!", "~", "-", "+", "*", "/", "%", "<", ">", "^", "|", "?"]
       
macros= ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",  # <stdio.h> macro
            "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
            "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]     # <stdlib.h> macro

special_ids = ["main",  # main function
                "stdio", "cstdio", "stdio.h",                                # <stdio.h> & <cstdio>
                "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",     # <stdio.h> types & streams
                "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush", # <stdio.h> functions
                "fopen", "freopen", "setbuf", "setvbuf", "fprintf", "fscanf",
                "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf",
                "vscanf", "vsnprintf", "vsprintf", "vsscanf", "fgetc", "fgets",
                "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc",
                "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell",
                "rewind", "clearerr", "feof", "ferror", "perror", "getline"
                "stdlib", "cstdlib", "stdlib.h",                             # <stdlib.h> & <cstdlib>
                "size_t", "div_t", "ldiv_t", "lldiv_t",                      # <stdlib.h> types
                "atof", "atoi", "atol", "atoll", "strtod", "strtof", "strtold",  # <stdlib.h> functions
                "strtol", "strtoll", "strtoul", "strtoull", "rand", "srand",
                "aligned_alloc", "calloc", "malloc", "realloc", "free", "abort",
                "atexit", "exit", "at_quick_exit", "_Exit", "getenv",
                "quick_exit", "system", "bsearch", "qsort", "abs", "labs",
                "llabs", "div", "ldiv", "lldiv", "mblen", "mbtowc", "wctomb",
                "mbstowcs", "wcstombs",
                "string", "cstring", "string.h",                                 # <string.h> & <cstring>
                "memcpy", "memmove", "memchr", "memcmp", "memset", "strcat",     # <string.h> functions
                "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll",
                "strcpy", "strncpy", "strerror", "strlen", "strspn", "strcspn",
                "strpbrk" ,"strstr", "strtok", "strxfrm",
                "memccpy", "mempcpy", "strcat_s", "strcpy_s", "strdup",      # <string.h> extension functions
                "strerror_r", "strlcat", "strlcpy", "strsignal", "strtok_r",
                "iostream", "istream", "ostream", "fstream", "sstream",      # <iostream> family
                "iomanip", "iosfwd",
                "ios", "wios", "streamoff", "streampos", "wstreampos",       # <iostream> types
                "streamsize", "cout", "cerr", "clog", "cin",
                "boolalpha", "noboolalpha", "skipws", "noskipws", "showbase",    # <iostream> manipulators
                "noshowbase", "showpoint", "noshowpoint", "showpos",
                "noshowpos", "unitbuf", "nounitbuf", "uppercase", "nouppercase",
                "left", "right", "internal", "dec", "oct", "hex", "fixed",
                "scientific", "hexfloat", "defaultfloat", "width", "fill",
                "precision", "endl", "ends", "flush", "ws", "showpoint",
                "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",    # <math.h> functions
                "cosh", "tanh", "exp", "sqrt", "log", "log10", "pow", "powf",
                "ceil", "floor", "abs", "fabs", "cabs", "frexp", "ldexp",
                "modf", "fmod", "hypot", "ldexp", "poly", "matherr"]

C_ast_key_words=['Alignas', 'ArrayDecl', 'ArrayRef', 'Assignment', 
                'BinaryOp', 'Break', 'Case', 'Cast', 'Compound', 
                'CompoundLiteral', 'Constant', 'Continue', 'Decl', 
                'DeclList', 'Default', 'DoWhile', 'EllipsisParam', 
                'EmptyStatement', 'Enum', 'Enumerator', 
                'EnumeratorList', 'ExprList', 'FileAST', 'For', 
                'FuncCall', 'FuncDecl', 'FuncDef', 'Goto', 'ID', 
                'IdentifierType', 'If', 'InitList', 'Label', 
                'NamedInitializer', 'ParamList', 'Pragma', 'PtrDecl', 'Return', 
                'StaticAssert', 'Struct', 'StructRef', 'Switch', 
                'TernaryOp', 'TypeDecl', 'Typedef', 'Typename', 
                'UnaryOp', 'Union', 'While']

forbidden_tokens = key_words+ ops + macros + special_ids+C_ast_key_words

