'''
Author: Zhang
Date: 2022-08-06
LastEditTime: 2022-08-07
Description: dead code that can attack code

'''
inserts = [
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

